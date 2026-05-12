from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.core.exceptions import IntegrationError
from app.schemas.faceless_video import (
    GeneratedSceneAnimation,
    SceneAnimationGenerationRequest,
    SceneAnimationGenerationResponse,
)
from app.utils.output_paths import output_url, stage_output_dir


class AIAnimationService:
    MODEL_REPOSITORIES = {
        "wan-i2v": "Wan-AI/Wan2.2-I2V-A14B",
        "wan2.2-i2v": "Wan-AI/Wan2.2-I2V-A14B",
        "ltx-video": "Lightricks/LTX-Video",
    }

    def __init__(
        self,
        *,
        huggingface_client,
        output_dir: str,
        model: str = "Wan-AI/Wan2.2-I2V-A14B",
        cache_dir: str = "assets/generated_animations",
        enabled: bool = False,
        num_frames: int = 81,
        inference_steps: int = 30,
        guidance_scale: float = 5.0,
    ) -> None:
        self.huggingface_client = huggingface_client
        self.output_dir = Path(output_dir)
        self.model = model
        self.cache_dir = self._resolve_cache_dir(cache_dir)
        self.enabled = enabled
        self.num_frames = num_frames
        self.inference_steps = inference_steps
        self.guidance_scale = guidance_scale

    def generate_scene_animations(self, payload: SceneAnimationGenerationRequest) -> SceneAnimationGenerationResponse:
        if not self.enabled:
            raise IntegrationError("AI animation generation is disabled. Set PY_WORKER_ENABLE_AI_ANIMATION=true.")
        if not self.huggingface_client.is_configured():
            raise IntegrationError("PY_WORKER_HF_TOKEN is required for AI animation generation.")

        stage_dir = stage_output_dir(
            output_dir=self.output_dir,
            output_bucket=payload.output_bucket,
            project_title=payload.project_title,
            project_id=payload.project_id,
            stage_name="animations",
        )
        stage_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        animations: list[GeneratedSceneAnimation] = []
        scenes_by_index = {scene.scene_index: scene for scene in payload.scenes}
        for image in payload.images:
            scene = scenes_by_index.get(image.scene_index)
            prompt = self.build_animation_prompt(
                scene_data=scene,
                image_prompt=image.prompt,
                animation_style=payload.animation_style,
            )
            cache_key = self._cache_key(
                image_path=image.image_path,
                prompt=prompt,
                model=self.model,
                num_frames=payload.num_frames or self.num_frames,
            )
            cached_path = self.cache_dir / f"{cache_key}.mp4"
            output_path = stage_dir / f"scene_{image.scene_index:02d}_animation.mp4"

            if cached_path.exists():
                output_path.write_bytes(cached_path.read_bytes())
            else:
                video_bytes = self.huggingface_client.image_to_video(
                    model=self._model_repository(),
                    image_path=image.image_path,
                    prompt=prompt,
                    negative_prompt="low quality, distorted, flicker, warped faces, text, subtitles, logos",
                    num_frames=payload.num_frames or self.num_frames,
                    num_inference_steps=payload.num_inference_steps or self.inference_steps,
                    guidance_scale=payload.guidance_scale or self.guidance_scale,
                    seed=self._seed_for_prompt(prompt),
                )
                cached_path.write_bytes(video_bytes)
                output_path.write_bytes(video_bytes)
                self._write_metadata(
                    metadata_path=cached_path.with_suffix(".json"),
                    prompt=prompt,
                    image_path=image.image_path,
                    cache_key=cache_key,
                )

            animations.append(
                GeneratedSceneAnimation(
                    scene_index=image.scene_index,
                    prompt=prompt,
                    source_image_path=image.image_path,
                    video_path=str(output_path.resolve()),
                    video_url=output_url(output_dir=self.output_dir, file_path=output_path),
                    cache_key=cache_key,
                )
            )

        return SceneAnimationGenerationResponse(
            job_id=payload.job_id,
            project_id=payload.project_id,
            animations=animations,
        )

    def build_animation_prompt(
        self,
        *,
        scene_data: Any,
        image_prompt: str | None,
        animation_style: str | None,
    ) -> str:
        narration = getattr(scene_data, "narration", "") if scene_data is not None else ""
        caption = getattr(scene_data, "caption_text", "") if scene_data is not None else ""
        style = animation_style or "cinematic story animation"
        prompt_parts = [
            style,
            image_prompt or "",
            narration,
            caption,
            "subtle realistic motion, cinematic camera movement, vertical story video",
            "preserve the subject identity and composition",
        ]
        return ", ".join(" ".join(part.split()) for part in prompt_parts if part)

    def _cache_key(self, *, image_path: str, prompt: str, model: str, num_frames: int) -> str:
        image_digest = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()[:16]
        normalized = json.dumps(
            {
                "image_digest": image_digest,
                "model": model,
                "num_frames": num_frames,
                "prompt": " ".join(prompt.lower().split()),
            },
            sort_keys=True,
        )
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]

    def _resolve_cache_dir(self, cache_dir: str) -> Path:
        path = Path(cache_dir)
        if path.is_absolute():
            return path
        return Path.cwd() / path

    def _model_repository(self) -> str:
        return self.MODEL_REPOSITORIES.get(self.model.lower(), self.model)

    def _seed_for_prompt(self, prompt: str) -> int:
        return int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)

    def _write_metadata(self, *, metadata_path: Path, prompt: str, image_path: str, cache_key: str) -> None:
        metadata_path.write_text(
            json.dumps(
                {
                    "cache_key": cache_key,
                    "model": self.model,
                    "image_path": image_path,
                    "prompt": prompt,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
