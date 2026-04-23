from pathlib import Path
import random
import shutil
import textwrap
import wave

from PIL import Image, ImageDraw, ImageFont

from app.core.exceptions import IntegrationError, ValidationError
from app.schemas.faceless_video import StoryRenderRequest, StoryRenderResponse
from app.utils.output_paths import output_url, stage_output_dir


class StoryRenderService:
    FONT_DIR = Path(__file__).resolve().parents[1] / "fonts" / "__pycache__"
    SCENE_CLIP_DIRNAME = "scene_clips"
    SCENE_FPS = 30
    SCENE_FADE_IN_SECONDS = 0.18
    SCENE_FADE_OUT_SECONDS = 0.28
    SCENE_START_ZOOM = 1.0
    SCENE_END_ZOOM = 1.06
    SCENE_WORK_WIDTH = 1280
    SCENE_WORK_HEIGHT = 2276
    MUSIC_MIN_START_OFFSET_SECONDS = 10.0
    MUSIC_MAX_START_OFFSET_SECONDS = 15.0
    REDDIT_TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "assets" / "template" / "title_template.png"
    REDDIT_CARD_VISIBLE_SECONDS = 4.2
    def __init__(
        self,
        ffmpeg_client,
        output_dir: str,
        music_service=None,
        llm_service=None,
        enable_background_music: bool = True,
        default_music_volume: float = 0.15,
        enable_audio_ducking: bool = True,
        reddit_story_background_video_path: str | None = None,
        reddit_story_background_music_path: str | None = None,
    ) -> None:
        self.ffmpeg_client = ffmpeg_client
        self.output_dir = Path(output_dir)
        self.music_service = music_service
        self.llm_service = llm_service
        self.enable_background_music = enable_background_music
        self.default_music_volume = default_music_volume
        self.enable_audio_ducking = enable_audio_ducking
        self.reddit_story_background_video_path = reddit_story_background_video_path
        self.reddit_story_background_music_path = reddit_story_background_music_path

    def render_story_video(self, payload: StoryRenderRequest) -> StoryRenderResponse:
        if not self.ffmpeg_client.is_available():
            raise IntegrationError("ffmpeg is required to render faceless story videos.")

        stage_dir = stage_output_dir(
            output_dir=self.output_dir,
            project_title=payload.project_title,
            project_id=payload.project_id,
            stage_name="render",
        )
        stage_dir.mkdir(parents=True, exist_ok=True)
        concat_path = stage_dir / "scene_inputs.txt"
        output_path = stage_dir / "faceless_story.mp4"
        render_audio_path = payload.audio_path
        scene_clip_dir = stage_dir / self.SCENE_CLIP_DIRNAME
        scene_clip_dir.mkdir(parents=True, exist_ok=True)
        audio_duration = self._audio_duration(Path(payload.audio_path))
        music_volume = payload.music_volume if payload.music_volume is not None else self.default_music_volume
        if self.enable_background_music and payload.use_music:
            selected_music_path = self._select_music_for_payload(payload)
            if selected_music_path:
                mixed_audio_path = stage_dir / "narration_with_music.wav"
                try:
                    self.mix_audio_with_music(
                        narration_path=Path(payload.audio_path),
                        music_path=Path(selected_music_path),
                        output_path=mixed_audio_path,
                        ducking=payload.ducking and self.enable_audio_ducking,
                        music_volume=music_volume,
                    )
                    render_audio_path = str(mixed_audio_path.resolve())
                except IntegrationError:
                    render_audio_path = payload.audio_path

        if payload.render_mode == "background_video":
            background_video_path = self._resolve_background_video_path(payload)
            total_duration = max(audio_duration or 0.0, 1.0)
            reddit_intro_card_path = self._build_reddit_intro_card(payload, stage_dir)
            render_subtitles_path = Path(payload.subtitles_path) if payload.subtitles_path else None
            if render_subtitles_path and render_subtitles_path.exists():
                safe_subtitle_dir = self.output_dir / "_render_tmp" / str(payload.project_id)
                safe_subtitle_dir.mkdir(parents=True, exist_ok=True)
                render_subtitles_path = self._copy_subtitles_to_safe_path(
                    subtitles_path=render_subtitles_path,
                    target_dir=safe_subtitle_dir,
                )
            self._render_background_video(
                background_video_path=background_video_path,
                audio_path=Path(render_audio_path),
                subtitles_path=render_subtitles_path,
                output_path=output_path,
                duration=total_duration,
                reddit_intro_card_path=reddit_intro_card_path,
            )
            return StoryRenderResponse(
                job_id=payload.job_id,
                project_id=payload.project_id,
                video_path=str(output_path.resolve()),
                video_url=output_url(output_dir=self.output_dir, file_path=output_path),
                duration_seconds=round(total_duration, 2),
            )

        if not payload.image_paths:
            raise ValidationError("At least one scene image is required for rendering.")

        subtitle_durations = None
        if payload.subtitles_path:
            subtitle_durations = self._duration_plan_from_subtitles(
                subtitles_path=Path(payload.subtitles_path),
                scene_count=len(payload.image_paths),
            )

        if subtitle_durations:
            planned_durations = subtitle_durations
        else:
            planned_durations = self._scene_duration_plan(
                payload=payload,
                audio_duration=audio_duration,
            )

        if audio_duration and planned_durations:
            planned_total = sum(planned_durations)
            if planned_total < audio_duration:
                planned_durations[-1] += audio_duration - planned_total

        scene_clip_paths: list[Path] = []
        concat_lines = []
        total_duration = 0.0
        for index, (image_path, duration) in enumerate(zip(payload.image_paths, planned_durations, strict=True), start=1):
            scene_clip_path = scene_clip_dir / f"scene_{index:02d}.mp4"
            self._render_scene_clip(
                image_path=Path(image_path),
                output_path=scene_clip_path,
                duration=duration,
            )
            scene_clip_paths.append(scene_clip_path)
            concat_lines.append(f"file '{scene_clip_path.resolve().as_posix()}'")
            total_duration += duration

        concat_path.write_text("\n".join(concat_lines), encoding="utf-8")

        filters = [
            "format=yuv420p",
        ]
        if payload.subtitles_path:
            subtitle_filter = f"subtitles='{self._escape_filter_path(Path(payload.subtitles_path))}'"
            if self.FONT_DIR.exists():
                subtitle_filter += f":fontsdir='{self._escape_filter_path(self.FONT_DIR)}'"
            filters.append(subtitle_filter)

        command = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-i",
            render_audio_path,
            "-vf",
            ",".join(filters),
            "-r",
            str(self.SCENE_FPS),
            "-t",
            str(max(total_duration, 1.0)),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-c:a",
            "aac",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        self.ffmpeg_client.run(command)

        return StoryRenderResponse(
            job_id=payload.job_id,
            project_id=payload.project_id,
            video_path=str(output_path.resolve()),
            video_url=output_url(output_dir=self.output_dir, file_path=output_path),
            duration_seconds=round(total_duration, 2),
        )

    def _render_background_video(
        self,
        *,
        background_video_path: Path,
        audio_path: Path,
        subtitles_path: Path | None,
        output_path: Path,
        duration: float,
        reddit_intro_card_path: Path | None = None,
    ) -> None:
        base_video_filter = [
            "scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos",
            "crop=1080:1920",
            "format=rgba",
        ]
        if subtitles_path:
            subtitle_filter = f"subtitles='{self._escape_filter_path(subtitles_path)}'"
            if self.FONT_DIR.exists():
                subtitle_filter += f":fontsdir='{self._escape_filter_path(self.FONT_DIR)}'"
            base_video_filter.append(subtitle_filter)

        command = [
            "ffmpeg",
            "-y",
            "-stream_loop",
            "-1",
            "-i",
            str(background_video_path),
            "-i",
            str(audio_path),
        ]

        filter_complex = [f"[0:v]{','.join(base_video_filter)}[basev]"]
        map_video_stream = "[basev]"

        if reddit_intro_card_path and reddit_intro_card_path.exists():
            command.extend(["-i", str(reddit_intro_card_path)])
            filter_complex.append(
                f"[basev][2:v]overlay=0:0:enable='between(t,0,{self.REDDIT_CARD_VISIBLE_SECONDS})'[intro]"
            )
            map_video_stream = "[intro]"

        command.extend(
            [
                "-filter_complex",
                ";".join(filter_complex),
                "-map",
                map_video_stream,
                "-map",
                "1:a",
                "-r",
                str(self.SCENE_FPS),
                "-t",
                str(duration),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
        )

        self.ffmpeg_client.run(command)

    def _render_scene_clip(self, *, image_path: Path, output_path: Path, duration: float) -> None:
        if duration <= 0:
            raise ValidationError("Scene duration must be greater than zero.")

        frame_count = max(int(round(duration * self.SCENE_FPS)), 1)
        progress_denominator = max(frame_count - 1, 1)
        fade_in = min(self.SCENE_FADE_IN_SECONDS, max(duration * 0.18, 0.06))
        fade_out = min(self.SCENE_FADE_OUT_SECONDS, max(duration * 0.22, 0.08))
        fade_out_start = max(duration - fade_out, 0.0)
        zoom_expression = (
            f"{self.SCENE_START_ZOOM}"
            f"+({self.SCENE_END_ZOOM - self.SCENE_START_ZOOM})*(n/{progress_denominator})"
        )

        scene_filter = (
            f"scale={self.SCENE_WORK_WIDTH}:{self.SCENE_WORK_HEIGHT}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={self.SCENE_WORK_WIDTH}:{self.SCENE_WORK_HEIGHT},"
            f"scale=w={self.SCENE_WORK_WIDTH}*({zoom_expression}):"
            f"h={self.SCENE_WORK_HEIGHT}*({zoom_expression}):"
            "eval=frame:flags=lanczos,"
            f"crop={self.SCENE_WORK_WIDTH}:{self.SCENE_WORK_HEIGHT}:(iw-{self.SCENE_WORK_WIDTH})/2:(ih-{self.SCENE_WORK_HEIGHT})/2,"
            "scale=1080:1920:flags=lanczos,"
            f"trim=duration={duration},"
            "setpts=PTS-STARTPTS,"
            f"fade=t=in:st=0:d={fade_in},"
            f"fade=t=out:st={fade_out_start}:d={fade_out},"
            "format=yuv420p"
        )

        self.ffmpeg_client.run(
            [
                "ffmpeg",
                "-y",
                "-loop",
                "1",
                "-i",
                str(image_path),
                "-vf",
                scene_filter,
                "-t",
                str(duration),
                "-an",
                "-r",
                str(self.SCENE_FPS),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                str(output_path),
            ]
        )

    def _scene_duration_plan(
        self,
        *,
        payload: StoryRenderRequest,
        audio_duration: float | None,
    ) -> list[float]:
        scene_total_duration = sum(
            max(scene.duration_seconds if scene else 5.0, 1.0)
            for scene in payload.scenes[: len(payload.image_paths)]
        )
        duration_scale = (
            audio_duration / scene_total_duration
            if audio_duration and scene_total_duration > 0
            else 1.0
        )
        planned_durations: list[float] = []
        for index, _image_path in enumerate(payload.image_paths):
            scene = payload.scenes[index] if index < len(payload.scenes) else None
            duration = max((scene.duration_seconds if scene else 5.0) * duration_scale, 1.0)
            planned_durations.append(duration)
        return planned_durations

    def _duration_plan_from_subtitles(self, subtitles_path: Path, scene_count: int) -> list[float] | None:
        if scene_count <= 0 or not subtitles_path.exists():
            return None

        cue_ranges: list[tuple[float, float]] = []
        suffix = subtitles_path.suffix.lower()
        if suffix == ".ass":
            cue_ranges = self._parse_ass_cues(subtitles_path)
        elif suffix == ".srt":
            cue_ranges = self._parse_srt_cues(subtitles_path)

        cue_ranges = [
            (max(start, 0.0), max(end, start + 0.1))
            for start, end in cue_ranges
            if end > start
        ]

        if not cue_ranges:
            return None

        # Word-by-word ASS karaoke tracks can contain dozens of short dialogue
        # events for a small number of scene images. Using those events to drive
        # scene durations shrinks the video below the narration length, so fall
        # back to audio-scaled scene timing in that case.
        if len(cue_ranges) > scene_count * 3:
            return None

        if len(cue_ranges) == scene_count:
            return [max(end - start, 0.1) for start, end in cue_ranges]

        if len(cue_ranges) < scene_count:
            return None

        base_group_size = len(cue_ranges) // scene_count
        remainder = len(cue_ranges) % scene_count

        durations: list[float] = []
        cursor = 0
        for index in range(scene_count):
            group_size = base_group_size + (1 if index < remainder else 0)
            group = cue_ranges[cursor : cursor + group_size]
            if not group:
                return None
            cursor += group_size

            start = group[0][0]
            end = group[-1][1]
            durations.append(max(end - start, 0.1))

        return durations

    def _parse_ass_cues(self, path: Path) -> list[tuple[float, float]]:
        cues: list[tuple[float, float]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("Dialogue:"):
                continue

            parts = line.split(",", 9)
            if len(parts) < 3:
                continue

            start = self._parse_ass_timestamp(parts[1].strip())
            end = self._parse_ass_timestamp(parts[2].strip())
            cues.append((start, end))

        return cues

    def _parse_srt_cues(self, path: Path) -> list[tuple[float, float]]:
        cues: list[tuple[float, float]] = []
        content = path.read_text(encoding="utf-8")
        blocks = [block.strip() for block in content.split("\n\n") if block.strip()]
        for block in blocks:
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            timeline = next((line for line in lines if " --> " in line), None)
            if not timeline:
                continue

            start_raw, end_raw = timeline.split(" --> ", 1)
            start = self._parse_srt_timestamp(start_raw)
            end = self._parse_srt_timestamp(end_raw)
            cues.append((start, end))

        return cues

    def _parse_ass_timestamp(self, value: str) -> float:
        # ASS format: h:mm:ss.cs
        parts = value.split(":")
        if len(parts) != 3:
            return 0.0

        hours = int(parts[0])
        minutes = int(parts[1])
        seconds_raw, centiseconds_raw = (parts[2].split(".", 1) + ["0"])[:2]
        seconds = int(seconds_raw)
        centiseconds = int(centiseconds_raw)

        return hours * 3600 + minutes * 60 + seconds + centiseconds / 100.0

    def _parse_srt_timestamp(self, value: str) -> float:
        # SRT format: HH:MM:SS,mmm
        parts = value.split(":")
        if len(parts) != 3:
            return 0.0

        hours = int(parts[0])
        minutes = int(parts[1])
        seconds_raw, millis_raw = (parts[2].split(",", 1) + ["0"])[:2]
        seconds = int(seconds_raw)
        milliseconds = int(millis_raw)

        return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0

    def _escape_filter_path(self, path: Path) -> str:
        escaped = path.resolve().as_posix().replace("\\", "/")
        escaped = escaped.replace(":", r"\:")
        escaped = escaped.replace("'", r"\'")
        return escaped

    def _audio_duration(self, audio_path: Path) -> float | None:
        try:
            with wave.open(str(audio_path), "rb") as audio_file:
                frame_count = audio_file.getnframes()
                frame_rate = audio_file.getframerate()
                if frame_rate <= 0:
                    return None
                return round(frame_count / float(frame_rate), 2)
        except (FileNotFoundError, wave.Error):
            return None

    def mix_audio_with_music(
        self,
        *,
        narration_path: Path,
        music_path: Path,
        output_path: Path,
        ducking: bool = True,
        music_volume: float = 0.15,
    ) -> None:
        if not music_path.exists():
            raise IntegrationError("Background music track was not found.")

        music_volume = min(max(music_volume, 0.0), 1.0)
        narration_duration = self._audio_duration(narration_path)
        if narration_duration is None:
            raise IntegrationError("Narration duration could not be determined for music mixing.")
        music_start_offset = round(
            random.uniform(self.MUSIC_MIN_START_OFFSET_SECONDS, self.MUSIC_MAX_START_OFFSET_SECONDS),
            2,
        )
        narration_filter = "[0:a]aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo[narr]"
        music_filter = (
            f"[1:a]aresample=44100,"
            "aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"atrim=start={music_start_offset}:duration={narration_duration},"
            f"asetpts=PTS-STARTPTS,volume={music_volume}[music]"
        )

        if ducking:
            filter_complex = (
                f"{narration_filter};"
                f"{music_filter};"
                "[music][narr]sidechaincompress=threshold=0.02:ratio=6:attack=20:release=250[ducked];"
                "[narr][ducked]amix=inputs=2:duration=first:dropout_transition=0,"
                "alimiter=limit=0.95[aout]"
            )
        else:
            filter_complex = (
                f"{narration_filter};"
                f"{music_filter};"
                "[narr][music]amix=inputs=2:duration=first:dropout_transition=0,"
                "alimiter=limit=0.95[aout]"
            )

        self.ffmpeg_client.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(narration_path),
                "-stream_loop",
                "-1",
                "-i",
                str(music_path),
                "-filter_complex",
                filter_complex,
                "-map",
                "[aout]",
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ]
        )

    def _select_music_for_payload(self, payload: StoryRenderRequest) -> str | None:
        if payload.background_music_path:
            explicit_path = Path(payload.background_music_path)
            if explicit_path.exists():
                return str(explicit_path.resolve())
            return None

        if payload.render_mode == "background_video" and self.reddit_story_background_music_path:
            configured_path = Path(self.reddit_story_background_music_path)
            if configured_path.exists():
                return str(configured_path.resolve())

        if self.music_service is None:
            return None

        script_text = " ".join(scene.narration for scene in payload.scenes if scene.narration).strip()
        detected_mood = self._detect_mood(script_text) if script_text else "neutral"
        return self.music_service.get_music_for_mood(detected_mood)

    def _detect_mood(self, script: str) -> str:
        if self.llm_service is not None and hasattr(self.llm_service, "detect_mood"):
            try:
                return self.llm_service.detect_mood(script)
            except Exception:
                pass
        if self.music_service is not None and hasattr(self.music_service, "detect_mood"):
            return self.music_service.detect_mood(script)
        return "neutral"

    def _resolve_background_video_path(self, payload: StoryRenderRequest) -> Path:
        if payload.background_video_path:
            explicit_path = Path(payload.background_video_path)
            if explicit_path.exists():
                return explicit_path.resolve()

        if self.reddit_story_background_video_path:
            configured_path = Path(self.reddit_story_background_video_path)
            if configured_path.exists():
                return configured_path.resolve()

        raise ValidationError("A Reddit story background video must be configured before rendering.")

    def _build_reddit_intro_card(self, payload: StoryRenderRequest, stage_dir: Path) -> Path | None:
        if payload.render_mode != "background_video" or not self.REDDIT_TEMPLATE_PATH.exists():
            return None

        title = payload.project_title or "Reddit Story"

        intro_card_path = stage_dir / "reddit_intro_card.png"
        template_image = Image.open(self.REDDIT_TEMPLATE_PATH).convert("RGBA")
        transparent_template = self._make_black_pixels_transparent(template_image)
        draw = ImageDraw.Draw(transparent_template)

        title_font = self._load_font(42, bold=True)
        meta_font = self._load_font(24, bold=False)

        title_box = (215, 875, 930, 1205)
        meta_position = (215, 800)

        subreddit = self._guess_subreddit_from_title(title)
        draw.text(meta_position, subreddit, fill=(36, 36, 36, 255), font=meta_font)
        self._draw_wrapped_text(
            draw,
            title_box,
            title,
            title_font,
            fill=(24, 24, 24, 255),
            line_spacing=8,
            max_lines=4,
        )

        transparent_template.save(intro_card_path)
        return intro_card_path

    def _make_black_pixels_transparent(self, image: Image.Image) -> Image.Image:
        converted = image.copy()
        pixels = converted.load()
        width, height = converted.size
        for x in range(width):
            for y in range(height):
                red, green, blue, alpha = pixels[x, y]
                if red <= 10 and green <= 10 and blue <= 10:
                    pixels[x, y] = (red, green, blue, 0)
                else:
                    pixels[x, y] = (red, green, blue, alpha)
        return converted

    def _load_font(self, size: int, *, bold: bool) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        preferred_fonts = ["DejaVuSans-Bold.ttf", "arialbd.ttf"] if bold else ["DejaVuSans.ttf", "arial.ttf"]
        for font_name in preferred_fonts:
            try:
                return ImageFont.truetype(font_name, size=size)
            except OSError:
                continue
        return ImageFont.load_default()

    def _draw_wrapped_text(
        self,
        draw: ImageDraw.ImageDraw,
        box: tuple[int, int, int, int],
        text: str,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        *,
        fill: tuple[int, int, int, int],
        line_spacing: int,
        max_lines: int | None = None,
    ) -> None:
        left, top, right, bottom = box
        max_width = right - left
        wrapped_lines = self._wrap_to_width(draw, text, font, max_width)
        if max_lines is not None and len(wrapped_lines) > max_lines:
            wrapped_lines = wrapped_lines[:max_lines]
            if wrapped_lines:
                wrapped_lines[-1] = self._ellipsis_to_width(draw, wrapped_lines[-1], font, max_width)

        y_cursor = top
        for line in wrapped_lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_height = bbox[3] - bbox[1]
            if y_cursor + line_height > bottom:
                break
            draw.text((left, y_cursor), line, fill=fill, font=font)
            y_cursor += line_height + line_spacing

    def _wrap_to_width(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        max_width: int,
    ) -> list[str]:
        words = text.split()
        if not words:
            return []

        lines: list[str] = []
        current_line = words[0]
        for word in words[1:]:
            candidate = f"{current_line} {word}"
            if self._text_width(draw, candidate, font) <= max_width:
                current_line = candidate
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)
        return lines

    def _ellipsis_to_width(
        self,
        draw: ImageDraw.ImageDraw,
        line: str,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        max_width: int,
    ) -> str:
        candidate = line.rstrip(". ") + "..."
        while candidate and self._text_width(draw, candidate, font) > max_width:
            candidate = candidate[:-4].rstrip() + "..."
        return candidate

    def _text_width(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    ) -> int:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0]

    def _shift_subtitles_for_intro(self, *, subtitles_path: Path, stage_dir: Path, offset_seconds: float) -> Path:
        suffix = subtitles_path.suffix.lower()
        shifted_path = stage_dir / f"{subtitles_path.stem}_intro_shifted{suffix}"

        if suffix == ".srt":
            shifted_lines: list[str] = []
            for line in subtitles_path.read_text(encoding="utf-8").splitlines():
                if " --> " not in line:
                    shifted_lines.append(line)
                    continue

                start_raw, end_raw = line.split(" --> ", 1)
                shifted_start = self._format_srt_timestamp(self._parse_srt_timestamp(start_raw) + offset_seconds)
                shifted_end = self._format_srt_timestamp(self._parse_srt_timestamp(end_raw) + offset_seconds)
                shifted_lines.append(f"{shifted_start} --> {shifted_end}")

            shifted_path.write_text("\n".join(shifted_lines), encoding="utf-8")
            return shifted_path

        if suffix == ".ass":
            shifted_lines: list[str] = []
            for line in subtitles_path.read_text(encoding="utf-8").splitlines():
                if not line.startswith("Dialogue:"):
                    shifted_lines.append(line)
                    continue

                parts = line.split(",", 9)
                if len(parts) < 10:
                    shifted_lines.append(line)
                    continue

                parts[1] = self._format_ass_timestamp(self._parse_ass_timestamp(parts[1].strip()) + offset_seconds)
                parts[2] = self._format_ass_timestamp(self._parse_ass_timestamp(parts[2].strip()) + offset_seconds)
                shifted_lines.append(",".join(parts))

            shifted_path.write_text("\n".join(shifted_lines), encoding="utf-8")
            return shifted_path

        return subtitles_path

    def _copy_subtitles_to_safe_path(self, *, subtitles_path: Path, target_dir: Path) -> Path:
        target_dir.mkdir(parents=True, exist_ok=True)
        safe_path = target_dir / subtitles_path.name
        shutil.copyfile(subtitles_path, safe_path)
        return safe_path

    def _format_ass_timestamp(self, seconds: float) -> str:
        total_centiseconds = max(int(round(seconds * 100)), 0)
        hours = total_centiseconds // 360000
        remaining = total_centiseconds % 360000
        minutes = remaining // 6000
        remaining %= 6000
        secs = remaining // 100
        centiseconds = remaining % 100
        return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"

    def _format_srt_timestamp(self, seconds: float) -> str:
        total_milliseconds = max(int(round(seconds * 1000)), 0)
        hours = total_milliseconds // 3600000
        remaining = total_milliseconds % 3600000
        minutes = remaining // 60000
        remaining %= 60000
        secs = remaining // 1000
        milliseconds = remaining % 1000
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    def _guess_subreddit_from_title(self, title: str) -> str:
        cleaned = textwrap.shorten(title.replace("\n", " "), width=28, placeholder="…")
        return f"Trending story • {cleaned}"
