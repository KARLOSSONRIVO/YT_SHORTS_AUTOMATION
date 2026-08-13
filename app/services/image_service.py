from io import BytesIO
from pathlib import Path
import hashlib
import json
import logging
import re

from PIL import Image

from app.core.exceptions import ContentSafetyError, IntegrationError
from app.schemas.faceless_video import (
    GeneratedSceneImage,
    SceneImageGenerationRequest,
    SceneImageGenerationResponse,
)
from app.services.cloudflare_image_generation_service import CloudflareImageGenerationService
from app.utils.output_paths import output_url, stage_output_dir

logger = logging.getLogger(__name__)


class ImageService:
    COLORS = ["0x22333B", "0x5E503F", "0x2D3142", "0x3A5A40", "0x6D597A", "0x355070"]
    TARGET_WIDTH = 1080
    TARGET_HEIGHT = 1920
    IMAGE_GENERATION_ATTEMPTS = 4
    IMAGE_NEGATIVE_PROMPT = (
        "text, words, letters, typography, captions, subtitles, closed captions, "
        "watermark, logo, ui overlay, lower third, ticker, poster text, quote text, "
        "headline, credits, score bug, scoreboard, labels, signs, banners, "
        "jersey text, jersey numbers, shirt text, uniform lettering, brand marks, "
        "newspaper text, book text, document text, handwriting, calligraphy, "
        "footer text, top text, bottom text, embedded subtitles, burned in subtitles, "
        "subtitle strip, caption strip, meme text, random glyphs, symbols"
    )

    def __init__(
        self,
        *,
        ffmpeg_client,
        output_dir: str,
        image_generation_service: CloudflareImageGenerationService,
        allow_placeholder_generation: bool = False,
    ) -> None:
        self.ffmpeg_client = ffmpeg_client
        self.output_dir = Path(output_dir)
        self.image_generation_service = image_generation_service
        self.allow_placeholder_generation = allow_placeholder_generation

    def generate_scene_images(
        self, payload: SceneImageGenerationRequest
    ) -> SceneImageGenerationResponse:
        stage_dir = stage_output_dir(
            output_dir=self.output_dir,
            output_bucket=payload.output_bucket,
            project_title=payload.project_title,
            project_id=payload.project_id,
            stage_name="scenes",
        )
        stage_dir.mkdir(parents=True, exist_ok=True)
        images: list[GeneratedSceneImage] = []

        for index, scene in enumerate(payload.scenes):
            output_path = stage_dir / f"scene_{scene.scene_index:02d}.png"
            prompt = self._compose_generation_prompt(
                visual_style=payload.visual_style,
                scene_prompt=scene.image_prompt,
            )
            if self._is_reusable_scene_image(output_path, prompt=prompt):
                logger.info(
                    "Reusing completed scene image %s for scene %s.",
                    output_path,
                    scene.scene_index,
                )
            else:
                self._generation_manifest_path(output_path).unlink(missing_ok=True)
                try:
                    output_path.write_bytes(self._generate_clean_scene_image(prompt))
                    self._write_generation_manifest(output_path, prompt=prompt)
                except Exception as exc:
                    if not self.allow_placeholder_generation:
                        if isinstance(exc, IntegrationError):
                            raise
                        raise IntegrationError(f"Image generation failed: {exc}") from exc
                    self._write_placeholder_image(index=index, output_path=output_path)

            images.append(
                GeneratedSceneImage(
                    scene_index=scene.scene_index,
                    prompt=prompt,
                    image_path=str(output_path.resolve()),
                    image_url=output_url(output_dir=self.output_dir, file_path=output_path),
                )
            )

        return SceneImageGenerationResponse(
            job_id=payload.job_id,
            project_id=payload.project_id,
            images=images,
        )

    def _compose_generation_prompt(self, *, visual_style: str, scene_prompt: str) -> str:
        normalized_prompt = re.sub(r"\s+", " ", scene_prompt).strip()
        normalized_style = re.sub(r"\s+", " ", visual_style).strip()
        realism_boost = (
            "photorealistic, anatomically correct body proportions, realistic hands and feet, "
            "natural facial features, believable motion freeze, clean composition, high detail"
        )
        anti_text_directive = (
            "absolutely no visible text anywhere in the image, no readable or unreadable letters, "
            "no subtitle bars, no lower thirds, no captions, no signage, no posters, no screens with text, "
            "no jersey names, no jersey numbers, no uniform lettering, no logos, no watermarks, "
            "no newspaper clippings, no documents, no screens, no scorebugs, no scoreboard overlays, "
            "no quote cards, no meme text, no closed-caption strip, no footer strip"
        )
        negative_constraints = (
            "no text, no words, no letters, no typography, no captions, no logos, no watermarks, "
            "no subtitles, no burned-in subtitles, no bottom caption strip, no top title strip, "
            "no scoreboard overlay, no UI, "
            "no duplicated subjects, no extra limbs, no distorted anatomy, no floating objects"
        )
        domain_boost = self._domain_specific_boost(normalized_prompt)

        parts = [
            normalized_style,
            normalized_prompt,
            domain_boost,
            realism_boost,
            "vertical 9:16 frame",
            anti_text_directive,
            negative_constraints,
        ]
        return ", ".join(part for part in parts if part)

    def _generate_clean_scene_image(self, prompt: str) -> bytes:
        last_candidate: bytes | None = None
        generation_prompt = prompt

        for attempt in range(self.IMAGE_GENERATION_ATTEMPTS):
            attempt_prompt = generation_prompt
            if attempt > 0:
                attempt_prompt = (
                    f"{generation_prompt}, clean cinematic still frame only, absolutely no text artifacts, "
                    "no letters, no captions, no subtitle strip, no footer text, no overlay graphics"
                )

            try:
                try:
                    generated_image = self._generate_image(attempt_prompt)
                except ContentSafetyError:
                    if generation_prompt != prompt:
                        raise
                    generation_prompt = self._content_safe_fallback_prompt(prompt)
                    logger.warning(
                        "Cloudflare rejected a scene prompt through content safety filtering; "
                        "retrying with a safe anonymous version of the same scene."
                    )
                    generated_image = self._generate_image(generation_prompt)
                candidate = self._post_process_generated_image(generated_image)
            except (IntegrationError, OSError, ValueError) as exc:
                if last_candidate is None:
                    raise
                logger.warning(
                    "Scene cleanup retry failed after a usable image was generated; "
                    "keeping the last candidate with safe framing instead. Error: %s",
                    exc,
                )
                return self._force_safe_text_free_framing(last_candidate)
            last_candidate = candidate

            if not self._processed_image_still_has_text_like_artifact(candidate):
                return candidate

            logger.warning(
                "Generated scene still appears to contain text-like artifacts; retrying image generation "
                "(attempt %s/%s).",
                attempt + 1,
                self.IMAGE_GENERATION_ATTEMPTS,
            )

        if last_candidate is None:
            raise IntegrationError("Image generation failed before producing an image candidate.")

        aggressively_cleaned = self._aggressively_crop_text_bands(last_candidate)
        if not self._processed_image_still_has_text_like_artifact(aggressively_cleaned):
            return aggressively_cleaned

        logger.warning(
            "Generated scene still appears to contain text-like artifacts after retries; "
            "applying final safe framing cleanup instead of failing the stage."
        )
        return self._force_safe_text_free_framing(last_candidate)

    def _content_safe_fallback_prompt(self, prompt: str) -> str:
        lowered = prompt.lower()

        if "baseball" in lowered:
            action = "focused before the next play"
            if any(token in lowered for token in ("hit", "swing", "bat", "home run")):
                action = "swinging a bat during a game"
            elif any(token in lowered for token in ("pitch", "mound", "strikeout")):
                action = "pitching from the mound during a game"
            return (
                "adult baseball player in a plain unbranded uniform, "
                f"{action}, realistic baseball stadium and softly blurred crowd, "
                "cinematic sports photography, unobstructed stadium backdrop, "
                "no celebrity likeness, no readable text, no logos, no names, no jersey numbers"
            )

        sports = {
            "basketball": "adult basketball player in a plain unbranded uniform on a realistic indoor court",
            "soccer": "adult soccer player in a plain unbranded uniform on a realistic outdoor pitch",
            "football": "adult American football player in plain unbranded protective gear on a realistic field",
            "boxing": "adult boxer wearing plain gloves in a realistic boxing ring",
            "mma": "adult martial artist in plain sportswear inside a realistic training arena",
        }
        for token, scene in sports.items():
            if token in lowered:
                return (
                    f"{scene}, cinematic sports photography, believable game action, "
                    "no celebrity likeness, no readable text, no logos, no names, no numbers, "
                    "unobstructed stadium or arena backdrop"
                )

        return (
            "adult person in a realistic everyday setting related to the story, "
            "natural facial features and believable body proportions, cinematic documentary photography, "
            "no celebrity likeness, no readable text, no logos, no signs, no screens, no watermark"
        )

    def _is_reusable_scene_image(self, output_path: Path, *, prompt: str) -> bool:
        if not output_path.is_file() or output_path.stat().st_size <= 0:
            return False

        try:
            with Image.open(output_path) as image:
                width, height = image.size
                image.verify()
            if width <= 0 or height <= 0:
                return False
        except (OSError, ValueError):
            logger.warning(
                "Existing scene image %s is invalid and will be regenerated.",
                output_path,
            )
            return False

        manifest_path = self._generation_manifest_path(output_path)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            logger.info(
                "Existing scene image %s has no valid generation manifest and will be regenerated.",
                output_path,
            )
            return False

        return manifest == self._generation_manifest(prompt)

    def _generation_manifest_path(self, output_path: Path) -> Path:
        return output_path.with_suffix(".generation.json")

    def _generation_manifest(self, prompt: str) -> dict[str, str | int]:
        model = str(getattr(self.image_generation_service, "model", "unknown"))
        return {
            "version": 1,
            "provider": "cloudflare_workers_ai",
            "model": model,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        }

    def _write_generation_manifest(self, output_path: Path, *, prompt: str) -> None:
        self._generation_manifest_path(output_path).write_text(
            json.dumps(self._generation_manifest(prompt), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _domain_specific_boost(self, prompt: str) -> str:
        lowered = prompt.lower()
        sports_terms = {
            "tennis": (
                "realistic professional tennis scene on a regulation grass tennis court, visible tennis net and court lines, "
                "correctly held tennis racket and regulation tennis ball, plausible tennis posture, each person has exactly "
                "two arms and exactly two legs, full-body subjects clearly separated with no overlapping limbs, "
                "plain tennis clothing with no readable names, numbers, or logos"
            ),
            "soccer": (
                "realistic association football scene, regulation soccer ball, believable stadium perspective, "
                "athlete in a plausible kicking or sprinting pose, correct goal or pitch context, "
                "plain uniforms with no readable names or numbers"
            ),
            "football": (
                "realistic American football scene, regulation field markings, believable tackle or run pose, "
                "correct protective gear, stadium action photo feel, plain uniforms with no readable names or numbers"
            ),
            "basketball": (
                "realistic basketball scene, correct court markings, believable dribble, layup, dunk, or defensive stance, "
                "arena sports photography look, plain uniforms with no readable names or numbers"
            ),
            "baseball": (
                "realistic baseball scene, accurate bat or glove use, believable pitching or batting pose, "
                "regulation field context, plain uniforms with no readable names or numbers"
            ),
            "boxing": (
                "realistic boxing scene, accurate gloves, ring ropes, believable punch or guard stance, "
                "sports photography lighting"
            ),
            "mma": (
                "realistic MMA fight scene, accurate cage or mat setting, believable guard or striking stance, "
                "anatomically plausible action"
            ),
        }

        for token, boost in sports_terms.items():
            if token in lowered:
                return boost

        if any(token in lowered for token in ("news anchor", "broadcast", "interview", "podcast")):
            return (
                "realistic documentary still frame, believable person placement, natural studio or interview environment, "
                "cinematic but grounded composition"
            )

        return "grounded cinematic still frame, realistic environment and subject placement"

    def _generate_image(self, prompt: str) -> bytes:
        return self.image_generation_service.generate_image(
            prompt=prompt,
            negative_prompt=self.IMAGE_NEGATIVE_PROMPT,
        ).image_bytes

    def _post_process_generated_image(self, image_bytes: bytes) -> bytes:
        with Image.open(BytesIO(image_bytes)) as source_image:
            image = source_image.convert("RGB")
            image = self._trim_banner_bands(image)
            image = self._trim_subtitle_like_footer(image)
            image = self._cover_resize(image, target_width=self.TARGET_WIDTH, target_height=self.TARGET_HEIGHT)

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def _processed_image_still_has_text_like_artifact(self, image_bytes: bytes) -> bool:
        with Image.open(BytesIO(image_bytes)) as source_image:
            image = source_image.convert("RGB")
            return self._edge_has_text_band(image, from_bottom=True) or self._edge_has_text_band(
                image, from_bottom=False
            )

    def _edge_has_text_band(self, image: Image.Image, *, from_bottom: bool) -> bool:
        width, height = image.size
        if width <= 0 or height <= 0:
            return False

        pixels = image.load()
        max_scan_rows = max(int(height * 0.22), 60)
        suspicious_rows = 0
        minimum_suspicious_rows = max(int(height * 0.014), 10)

        if from_bottom:
            row_iterable = range(height - 1, max(height - max_scan_rows - 1, -1), -1)
        else:
            row_iterable = range(0, min(max_scan_rows, height))

        for y in row_iterable:
            bright_pixels = 0
            dark_pixels = 0
            sampled_pixels = 0
            for x in range(0, width, max(width // 180, 1)):
                red, green, blue = pixels[x, y]
                sampled_pixels += 1
                if red >= 190 and green >= 190 and blue >= 190:
                    bright_pixels += 1
                if red <= 75 and green <= 75 and blue <= 75:
                    dark_pixels += 1

            if sampled_pixels == 0:
                continue

            bright_ratio = bright_pixels / sampled_pixels
            dark_ratio = dark_pixels / sampled_pixels
            looks_like_text_band = dark_ratio >= 0.28 and bright_ratio >= 0.045
            looks_like_bright_footer = bright_ratio >= 0.52

            if looks_like_text_band or looks_like_bright_footer:
                suspicious_rows += 1

        return suspicious_rows >= minimum_suspicious_rows

    def _aggressively_crop_text_bands(self, image_bytes: bytes) -> bytes:
        with Image.open(BytesIO(image_bytes)) as source_image:
            image = source_image.convert("RGB")
            width, height = image.size
            top_crop = int(height * 0.05) if self._edge_has_text_band(image, from_bottom=False) else 0
            bottom_crop = int(height * 0.12) if self._edge_has_text_band(image, from_bottom=True) else 0
            cropped_bottom = max(height - bottom_crop, top_crop + 100)
            image = image.crop((0, top_crop, width, cropped_bottom))
            image = self._cover_resize(image, target_width=self.TARGET_WIDTH, target_height=self.TARGET_HEIGHT)

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def _force_safe_text_free_framing(self, image_bytes: bytes) -> bytes:
        with Image.open(BytesIO(image_bytes)) as source_image:
            image = source_image.convert("RGB")
            width, height = image.size

            top_crop = int(height * 0.07)
            bottom_crop = int(height * 0.18)
            side_crop = int(width * 0.02)

            left = min(side_crop, max(width // 10, 1))
            top = min(top_crop, max(height // 5, 1))
            right = max(width - side_crop, left + 100)
            bottom = max(height - bottom_crop, top + 100)

            image = image.crop((left, top, right, bottom))
            image = self._cover_resize(
                image,
                target_width=self.TARGET_WIDTH,
                target_height=self.TARGET_HEIGHT,
            )

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def _trim_banner_bands(self, image: Image.Image) -> Image.Image:
        image = self._trim_edge_banner(image, from_bottom=True)
        image = self._trim_edge_banner(image, from_bottom=False)
        return image

    def _trim_subtitle_like_footer(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width <= 0 or height <= 0:
            return image

        pixels = image.load()
        scan_start = int(height * 0.78)
        suspicious_rows = 0
        first_suspicious_row: int | None = None

        for y in range(scan_start, height):
            bright_pixels = 0
            dark_pixels = 0
            sampled_pixels = 0
            for x in range(0, width, max(width // 180, 1)):
                red, green, blue = pixels[x, y]
                sampled_pixels += 1
                if red >= 180 and green >= 180 and blue >= 180:
                    bright_pixels += 1
                if red <= 70 and green <= 70 and blue <= 70:
                    dark_pixels += 1

            if sampled_pixels == 0:
                continue

            bright_ratio = bright_pixels / sampled_pixels
            dark_ratio = dark_pixels / sampled_pixels

            if dark_ratio >= 0.45 and bright_ratio >= 0.08:
                suspicious_rows += 1
                if first_suspicious_row is None:
                    first_suspicious_row = y

        minimum_suspicious_rows = max(int(height * 0.025), 18)
        if first_suspicious_row is None or suspicious_rows < minimum_suspicious_rows:
            return image

        cropped_height = max(first_suspicious_row - 8, int(height * 0.72))
        return image.crop((0, 0, width, cropped_height))

    def _trim_edge_banner(self, image: Image.Image, *, from_bottom: bool) -> Image.Image:
        width, height = image.size
        if width <= 0 or height <= 0:
            return image

        pixels = image.load()
        band_start: int | None = None
        contiguous_rows = 0
        minimum_band_rows = max(int(height * 0.04), 24)
        max_band_rows = int(height * 0.3)

        if from_bottom:
            row_iterable = range(height - 1, max(height - max_band_rows - 1, -1), -1)
        else:
            row_iterable = range(0, min(max_band_rows, height))

        for y in row_iterable:
            bright_pixels = 0
            dark_pixels = 0
            sampled_pixels = 0
            for x in range(0, width, max(width // 160, 1)):
                red, green, blue = pixels[x, y]
                sampled_pixels += 1
                if red >= 235 and green >= 235 and blue >= 235:
                    bright_pixels += 1
                if red <= 45 and green <= 45 and blue <= 45:
                    dark_pixels += 1

            if sampled_pixels == 0:
                continue

            bright_ratio = bright_pixels / sampled_pixels
            dark_ratio = dark_pixels / sampled_pixels
            looks_like_bright_banner = bright_ratio >= 0.72
            looks_like_dark_banner_with_text = dark_ratio >= 0.72 and bright_ratio >= 0.01

            if looks_like_bright_banner or looks_like_dark_banner_with_text:
                band_start = y
                contiguous_rows += 1
                continue

            if contiguous_rows >= minimum_band_rows:
                break

            band_start = None
            contiguous_rows = 0

        if band_start is None or contiguous_rows < minimum_band_rows:
            return image

        if from_bottom:
            cropped_height = max(band_start, int(height * 0.6))
            return image.crop((0, 0, width, cropped_height))

        cropped_top = min(band_start + contiguous_rows, int(height * 0.18))
        return image.crop((0, cropped_top, width, height))

    def _cover_resize(self, image: Image.Image, *, target_width: int, target_height: int) -> Image.Image:
        source_width, source_height = image.size
        if source_width <= 0 or source_height <= 0:
            return image

        scale = max(target_width / source_width, target_height / source_height)
        resized_width = max(int(round(source_width * scale)), target_width)
        resized_height = max(int(round(source_height * scale)), target_height)
        resized = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

        left = max((resized_width - target_width) // 2, 0)
        top = max((resized_height - target_height) // 2, 0)
        return resized.crop((left, top, left + target_width, top + target_height))

    # ------------------------------------------------------------------ #
    # Placeholder fallback                                                 #
    # ------------------------------------------------------------------ #

    def _write_placeholder_image(self, *, index: int, output_path: Path) -> None:
        if not self.ffmpeg_client.is_available():
            raise IntegrationError("ffmpeg is required to create placeholder scene images.")

        color = self.COLORS[index % len(self.COLORS)]
        self.ffmpeg_client.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"color=c={color}:s=1080x1920",
                "-frames:v",
                "1",
                str(output_path),
            ]
        )
