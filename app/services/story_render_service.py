from pathlib import Path
import random
import shutil
import subprocess
import wave

from app.core.exceptions import IntegrationError, ValidationError
from app.schemas.faceless_video import StoryRenderRequest, StoryRenderResponse
from app.utils.output_paths import dated_stage_output_dir, output_url


class StoryRenderService:
    FONT_DIR = Path(__file__).resolve().parents[1] / "fonts" / "__pycache__"
    SCENE_CLIP_DIRNAME = "scene_clips"
    SCENE_FPS = 30
    SCENE_FADE_IN_SECONDS = 0.18
    SCENE_FADE_OUT_SECONDS = 0.28
    FIRST_SCENE_FADE_IN_SECONDS = 0.0
    SCENE_START_ZOOM = 1.0
    SCENE_END_ZOOM = 1.06
    SCENE_WORK_WIDTH = 1280
    SCENE_WORK_HEIGHT = 2276
    MUSIC_MIN_START_OFFSET_SECONDS = 10.0
    MUSIC_MAX_START_OFFSET_SECONDS = 15.0
    REDDIT_BACKGROUND_VIDEO_DIR = Path(__file__).resolve().parents[1] / "assets" / "video"
    REDDIT_BACKGROUND_VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv"}
    def __init__(
        self,
        ffmpeg_client,
        output_dir: str,
        music_service=None,
        llm_service=None,
        enable_background_music: bool = True,
        default_music_volume: float = 0.15,
        enable_audio_ducking: bool = True,
        reddit_story_background_video_dir: str | Path | None = None,
    ) -> None:
        self.ffmpeg_client = ffmpeg_client
        self.output_dir = Path(output_dir)
        self.music_service = music_service
        self.llm_service = llm_service
        self.enable_background_music = enable_background_music
        self.default_music_volume = default_music_volume
        self.enable_audio_ducking = enable_audio_ducking
        self.reddit_story_background_video_dir = Path(reddit_story_background_video_dir or self.REDDIT_BACKGROUND_VIDEO_DIR)

    def render_story_video(self, payload: StoryRenderRequest) -> StoryRenderResponse:
        if not self.ffmpeg_client.is_available():
            raise IntegrationError("ffmpeg is required to render faceless story videos.")

        stage_dir = dated_stage_output_dir(
            output_dir=self.output_dir,
            output_bucket=payload.output_bucket,
            project_title=payload.project_title,
            project_id=payload.project_id,
            stage_name="rendered_video",
        )
        stage_dir.mkdir(parents=True, exist_ok=True)
        concat_path = stage_dir / "scene_inputs.txt"
        output_path = stage_dir / "faceless_story.mp4"
        render_audio_path = payload.audio_path
        scene_clip_dir = stage_dir / self.SCENE_CLIP_DIRNAME
        scene_clip_dir.mkdir(parents=True, exist_ok=True)
        audio_duration = self._audio_duration(Path(payload.audio_path))
        music_volume = payload.music_volume if payload.music_volume is not None else self.default_music_volume
        narration_volume = payload.narration_volume if payload.narration_volume is not None else 1.0
        ambience_audio_paths = self._resolve_optional_audio_paths(getattr(payload, "ambience_audio_paths", []))
        sfx_audio_paths = self._resolve_optional_audio_paths(getattr(payload, "sfx_audio_paths", []))
        if ambience_audio_paths or sfx_audio_paths:
            selected_music_path = self._select_music_for_payload(payload) if self.enable_background_music and payload.use_music else None
            mixed_audio_path = stage_dir / "narration_with_cinematic_layers.wav"
            try:
                self.mix_audio_with_cinematic_layers(
                    narration_path=Path(payload.audio_path),
                    output_path=mixed_audio_path,
                    music_path=Path(selected_music_path) if selected_music_path else None,
                    ambience_paths=ambience_audio_paths,
                    sfx_paths=sfx_audio_paths,
                    ducking=payload.ducking and self.enable_audio_ducking,
                    music_volume=music_volume,
                    ambience_volume=getattr(payload, "ambience_volume", 0.08),
                    narration_volume=narration_volume,
                )
                render_audio_path = str(mixed_audio_path.resolve())
            except IntegrationError:
                render_audio_path = payload.audio_path
        elif self.enable_background_music and payload.use_music:
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
                        narration_volume=narration_volume,
                    )
                    render_audio_path = str(mixed_audio_path.resolve())
                except IntegrationError:
                    render_audio_path = payload.audio_path

        if payload.render_mode == "background_video":
            background_video_path = self._select_reddit_background_video()
            total_duration = max(audio_duration or 0.0, 1.0)
            output_path = stage_dir / "faceless_story.mp4"
            self._render_background_video(
                background_video_path=background_video_path,
                audio_path=Path(render_audio_path),
                subtitles_path=Path(payload.subtitles_path) if payload.subtitles_path else None,
                output_path=output_path,
                duration=total_duration,
            )
            return StoryRenderResponse(
                job_id=payload.job_id,
                project_id=payload.project_id,
                video_path=str(output_path.resolve()),
                video_url=output_url(output_dir=self.output_dir, file_path=output_path),
                duration_seconds=round(total_duration, 2),
            )

        scene_video_paths = self._resolve_optional_video_paths(getattr(payload, "scene_video_paths", []))
        if payload.render_mode == "animation_story" and not scene_video_paths and not payload.image_paths:
            raise ValidationError("At least one scene animation or scene image is required for animation rendering.")
        if payload.render_mode != "animation_story" and not payload.image_paths:
            raise ValidationError("At least one scene image is required for rendering.")

        render_source_count = len(scene_video_paths) if payload.render_mode == "animation_story" and scene_video_paths else len(payload.image_paths)
        subtitle_durations = None
        if payload.subtitles_path:
            subtitle_durations = self._duration_plan_from_subtitles(
                subtitles_path=Path(payload.subtitles_path),
                scene_count=render_source_count,
            )

        if subtitle_durations:
            planned_durations = subtitle_durations
        else:
            planned_durations = self._scene_duration_plan(
                payload=payload,
                audio_duration=audio_duration,
                scene_count=render_source_count,
            )

        if audio_duration and planned_durations:
            planned_total = sum(planned_durations)
            if planned_total < audio_duration:
                planned_durations[-1] += audio_duration - planned_total

        scene_clip_paths: list[Path] = []
        concat_lines = []
        total_duration = 0.0
        use_animated_scenes = payload.render_mode in {"animation_story", "animated_scene_images"}
        scene_sources = scene_video_paths if payload.render_mode == "animation_story" and scene_video_paths else payload.image_paths
        for index, (scene_source, duration) in enumerate(zip(scene_sources, planned_durations, strict=True), start=1):
            scene_clip_path = scene_clip_dir / f"scene_{index:02d}.mp4"
            if payload.render_mode == "animation_story" and scene_video_paths:
                self._render_existing_scene_video_clip(
                    video_path=Path(scene_source),
                    output_path=scene_clip_path,
                    duration=duration,
                    is_first_scene=index == 1,
                )
            elif use_animated_scenes:
                self._render_animated_scene_clip(
                    image_path=Path(scene_source),
                    output_path=scene_clip_path,
                    duration=duration,
                    scene_index=index,
                    is_first_scene=index == 1,
                    animation_style=payload.animation_style,
                    animation_intensity=payload.animation_intensity,
                )
            else:
                self._render_scene_clip(
                    image_path=Path(scene_source),
                    output_path=scene_clip_path,
                    duration=duration,
                    is_first_scene=index == 1,
                )
            scene_clip_paths.append(scene_clip_path)
            concat_lines.append(self._concat_file_entry(scene_clip_path))
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
    ) -> None:
        background_video_duration = self._media_duration(background_video_path)
        background_start_offset = self._background_video_start_offset(
            video_duration=background_video_duration,
            target_duration=duration,
        )
        base_video_filter = [
            "setpts=PTS-STARTPTS",
            "scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos",
            "crop=1080:1920",
            "format=yuv420p",
        ]
        if subtitles_path:
            subtitle_filter = f"subtitles='{self._escape_filter_path(subtitles_path)}'"
            if self.FONT_DIR.exists():
                subtitle_filter += f":fontsdir='{self._escape_filter_path(self.FONT_DIR)}'"
            base_video_filter.append(subtitle_filter)

        command = ["ffmpeg", "-y"]
        if background_start_offset > 0:
            command.extend(["-ss", str(round(background_start_offset, 2))])
        command.extend(
            [
                "-stream_loop",
                "-1",
                "-i",
                str(background_video_path),
                "-i",
                str(audio_path),
                "-vf",
                ",".join(base_video_filter),
                "-map",
                "0:v",
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
                "-c:a",
                "aac",
                "-shortest",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
        )
        self.ffmpeg_client.run(command)

    def _render_scene_clip(self, *, image_path: Path, output_path: Path, duration: float, is_first_scene: bool = False) -> None:
        if duration <= 0:
            raise ValidationError("Scene duration must be greater than zero.")

        frame_count = max(int(round(duration * self.SCENE_FPS)), 1)
        progress_denominator = max(frame_count - 1, 1)
        fade_in = self.FIRST_SCENE_FADE_IN_SECONDS if is_first_scene else min(self.SCENE_FADE_IN_SECONDS, max(duration * 0.18, 0.06))
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
            f"{'' if fade_in <= 0 else f'fade=t=in:st=0:d={fade_in},'}"
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

    def _render_animated_scene_clip(
        self,
        *,
        image_path: Path,
        output_path: Path,
        duration: float,
        scene_index: int,
        is_first_scene: bool = False,
        animation_style: str | None = None,
        animation_intensity: float = 1.0,
    ) -> None:
        if duration <= 0:
            raise ValidationError("Scene duration must be greater than zero.")

        frame_count = max(int(round(duration * self.SCENE_FPS)), 1)
        progress_denominator = max(frame_count - 1, 1)
        intensity = min(max(animation_intensity, 0.25), 2.0)
        style = (animation_style or "cinematic").strip().lower()
        fade_in = self.FIRST_SCENE_FADE_IN_SECONDS if is_first_scene else min(self.SCENE_FADE_IN_SECONDS, max(duration * 0.18, 0.06))
        fade_out = min(self.SCENE_FADE_OUT_SECONDS, max(duration * 0.22, 0.08))
        fade_out_start = max(duration - fade_out, 0.0)

        zoom_start = 1.03
        zoom_end = 1.11 + (0.035 * intensity)
        if "slow" in style or "subtle" in style:
            zoom_end = 1.08 + (0.02 * intensity)
        if "dramatic" in style or "action" in style:
            zoom_end = 1.16 + (0.04 * intensity)

        zoom_expression = f"{zoom_start}+({zoom_end - zoom_start})*(n/{progress_denominator})"
        pan_distance = max(int(42 * intensity), 12)
        pan_direction = scene_index % 4
        if pan_direction == 0:
            x_expression = f"(iw-{self.SCENE_WORK_WIDTH})/2+{pan_distance}*(n/{progress_denominator})"
            y_expression = f"(ih-{self.SCENE_WORK_HEIGHT})/2"
        elif pan_direction == 1:
            x_expression = f"(iw-{self.SCENE_WORK_WIDTH})/2-{pan_distance}*(n/{progress_denominator})"
            y_expression = f"(ih-{self.SCENE_WORK_HEIGHT})/2"
        elif pan_direction == 2:
            x_expression = f"(iw-{self.SCENE_WORK_WIDTH})/2"
            y_expression = f"(ih-{self.SCENE_WORK_HEIGHT})/2+{pan_distance}*(n/{progress_denominator})"
        else:
            x_expression = f"(iw-{self.SCENE_WORK_WIDTH})/2"
            y_expression = f"(ih-{self.SCENE_WORK_HEIGHT})/2-{pan_distance}*(n/{progress_denominator})"

        scene_filter = (
            f"scale={self.SCENE_WORK_WIDTH + pan_distance * 4}:{self.SCENE_WORK_HEIGHT + pan_distance * 4}:"
            "force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={self.SCENE_WORK_WIDTH + pan_distance * 2}:{self.SCENE_WORK_HEIGHT + pan_distance * 2},"
            f"scale=w={self.SCENE_WORK_WIDTH + pan_distance * 2}*({zoom_expression}):"
            f"h={self.SCENE_WORK_HEIGHT + pan_distance * 2}*({zoom_expression}):"
            "eval=frame:flags=lanczos,"
            f"crop={self.SCENE_WORK_WIDTH}:{self.SCENE_WORK_HEIGHT}:{x_expression}:{y_expression},"
            "scale=1080:1920:flags=lanczos,"
            f"trim=duration={duration},"
            "setpts=PTS-STARTPTS,"
            f"{'' if fade_in <= 0 else f'fade=t=in:st=0:d={fade_in},'}"
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

    def _render_existing_scene_video_clip(self, *, video_path: Path, output_path: Path, duration: float, is_first_scene: bool = False) -> None:
        if duration <= 0:
            raise ValidationError("Scene duration must be greater than zero.")

        fade_in = self.FIRST_SCENE_FADE_IN_SECONDS if is_first_scene else min(self.SCENE_FADE_IN_SECONDS, max(duration * 0.18, 0.06))
        fade_out = min(self.SCENE_FADE_OUT_SECONDS, max(duration * 0.22, 0.08))
        fade_out_start = max(duration - fade_out, 0.0)
        video_filter = (
            "setpts=PTS-STARTPTS,"
            "scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,"
            "crop=1080:1920,"
            f"trim=duration={duration},"
            "setpts=PTS-STARTPTS,"
            f"{'' if fade_in <= 0 else f'fade=t=in:st=0:d={fade_in},'}"
            f"fade=t=out:st={fade_out_start}:d={fade_out},"
            "format=yuv420p"
        )

        self.ffmpeg_client.run(
            [
                "ffmpeg",
                "-y",
                "-stream_loop",
                "-1",
                "-i",
                str(video_path),
                "-vf",
                video_filter,
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
        scene_count: int | None = None,
    ) -> list[float]:
        source_count = scene_count if scene_count is not None else len(payload.image_paths)
        scene_total_duration = sum(
            max(scene.duration_seconds if scene else 5.0, 1.0)
            for scene in payload.scenes[:source_count]
        )
        duration_scale = (
            audio_duration / scene_total_duration
            if audio_duration and scene_total_duration > 0
            else 1.0
        )
        planned_durations: list[float] = []
        for index in range(source_count):
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

    @staticmethod
    def _concat_file_entry(path: Path) -> str:
        normalized_path = path.resolve().as_posix()
        escaped_path = normalized_path.replace("'", r"'\''")
        return f"file '{escaped_path}'"

    def _escape_filter_path(self, path: Path) -> str:
        escaped = path.resolve().as_posix().replace("\\", "/")
        escaped = escaped.replace(":", r"\:")
        escaped = escaped.replace("'", r"'\\\''")
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

    def _select_reddit_background_video(self) -> Path:
        if not self.reddit_story_background_video_dir.exists():
            raise ValidationError("A Reddit story background video must be available in app/assets/video.")

        candidates = sorted(
            path.resolve()
            for path in self.reddit_story_background_video_dir.iterdir()
            if path.is_file() and path.suffix.lower() in self.REDDIT_BACKGROUND_VIDEO_EXTENSIONS
        )
        if not candidates:
            raise ValidationError("A Reddit story background video must be available in app/assets/video.")
        return random.choice(candidates)

    def _media_duration(self, media_path: Path) -> float | None:
        if not shutil.which("ffprobe") or not media_path.exists():
            return None

        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(media_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None

        try:
            duration = float(result.stdout.strip())
        except ValueError:
            return None
        return duration if duration > 0 else None

    def _background_video_start_offset(self, *, video_duration: float | None, target_duration: float) -> float:
        if video_duration is None or video_duration <= target_duration + 0.5:
            if video_duration is None or video_duration <= 8:
                return 0.0
            return random.uniform(0.0, max(video_duration - 6.0, 0.0))

        max_offset = max(video_duration - target_duration, 0.0)
        if max_offset <= 0:
            return 0.0
        return random.uniform(0.0, max_offset)

    def mix_audio_with_music(
        self,
        *,
        narration_path: Path,
        music_path: Path,
        output_path: Path,
        ducking: bool = True,
        music_volume: float = 0.15,
        narration_volume: float = 1.0,
    ) -> None:
        if not music_path.exists():
            raise IntegrationError("Background music track was not found.")

        music_volume = min(max(music_volume, 0.0), 1.0)
        narration_volume = min(max(narration_volume, 0.0), 2.0)
        narration_duration = self._audio_duration(narration_path)
        if narration_duration is None:
            raise IntegrationError("Narration duration could not be determined for music mixing.")
        music_start_offset = round(
            random.uniform(self.MUSIC_MIN_START_OFFSET_SECONDS, self.MUSIC_MAX_START_OFFSET_SECONDS),
            2,
        )
        narration_filter = (
            "[0:a]aresample=44100,"
            "aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"volume={narration_volume}[narr]"
        )
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

        script_text = " ".join(scene.narration for scene in payload.scenes if scene.narration).strip()
        detected_mood = self._detect_mood(script_text) if script_text else "neutral"
        if self.music_service is not None:
            selected_track = self.music_service.get_music_for_mood(detected_mood)
            if selected_track:
                return selected_track

        return None

    def _detect_mood(self, script: str) -> str:
        if self.llm_service is not None and hasattr(self.llm_service, "detect_mood"):
            try:
                return self.llm_service.detect_mood(script)
            except Exception:
                pass
        if self.music_service is not None and hasattr(self.music_service, "detect_mood"):
            return self.music_service.detect_mood(script)
        return "neutral"

    def _resolve_optional_audio_paths(self, paths: list[str] | None) -> list[Path]:
        resolved_paths: list[Path] = []
        for raw_path in paths or []:
            path = Path(raw_path)
            if path.exists():
                resolved_paths.append(path.resolve())
        return resolved_paths

    def _resolve_optional_video_paths(self, paths: list[str] | None) -> list[Path]:
        resolved_paths: list[Path] = []
        for raw_path in paths or []:
            path = Path(raw_path)
            if path.exists():
                resolved_paths.append(path.resolve())
        return resolved_paths

    def mix_audio_with_cinematic_layers(
        self,
        *,
        narration_path: Path,
        output_path: Path,
        music_path: Path | None = None,
        ambience_paths: list[Path] | None = None,
        sfx_paths: list[Path] | None = None,
        ducking: bool = True,
        music_volume: float = 0.15,
        ambience_volume: float = 0.08,
        narration_volume: float = 1.0,
        fade_seconds: float = 1.5,
    ) -> None:
        narration_duration = self._audio_duration(narration_path)
        if narration_duration is None:
            raise IntegrationError("Narration duration could not be determined for ambience mixing.")

        music_volume = min(max(music_volume, 0.0), 1.0)
        ambience_volume = min(max(ambience_volume, 0.0), 1.0)
        narration_volume = min(max(narration_volume, 0.0), 2.0)
        ambience_paths = ambience_paths or []
        sfx_paths = sfx_paths or []

        command = ["ffmpeg", "-y", "-i", str(narration_path)]
        filter_parts = [
            "[0:a]aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"volume={narration_volume}[narr]"
        ]
        bed_labels: list[str] = []
        input_index = 1

        if music_path and music_path.exists():
            music_start_offset = round(
                random.uniform(self.MUSIC_MIN_START_OFFSET_SECONDS, self.MUSIC_MAX_START_OFFSET_SECONDS),
                2,
            )
            command.extend(["-stream_loop", "-1", "-i", str(music_path)])
            filter_parts.append(
                f"[{input_index}:a]aresample=44100,"
                "aformat=sample_fmts=fltp:channel_layouts=stereo,"
                f"atrim=start={music_start_offset}:duration={narration_duration},"
                f"asetpts=PTS-STARTPTS,volume={music_volume}[music]"
            )
            bed_labels.append("[music]")
            input_index += 1

        fade_out_start = max(narration_duration - fade_seconds, 0.0)
        for ambience_index, ambience_path in enumerate(ambience_paths):
            command.extend(["-stream_loop", "-1", "-i", str(ambience_path)])
            label = f"amb{ambience_index}"
            filter_parts.append(
                f"[{input_index}:a]aresample=44100,"
                "aformat=sample_fmts=fltp:channel_layouts=stereo,"
                f"atrim=duration={narration_duration},asetpts=PTS-STARTPTS,"
                f"afade=t=in:st=0:d={fade_seconds},"
                f"afade=t=out:st={fade_out_start}:d={fade_seconds},"
                f"volume={ambience_volume}[{label}]"
            )
            bed_labels.append(f"[{label}]")
            input_index += 1

        for sfx_index, sfx_path in enumerate(sfx_paths):
            command.extend(["-i", str(sfx_path)])
            label = f"sfx{sfx_index}"
            filter_parts.append(
                f"[{input_index}:a]aresample=44100,"
                "aformat=sample_fmts=fltp:channel_layouts=stereo,"
                f"atrim=duration={narration_duration},asetpts=PTS-STARTPTS[{label}]"
            )
            bed_labels.append(f"[{label}]")
            input_index += 1

        if bed_labels:
            filter_parts.append(
                "".join(bed_labels)
                + f"amix=inputs={len(bed_labels)}:duration=first:dropout_transition=0[bed]"
            )
            if ducking:
                filter_parts.append(
                    "[bed][narr]sidechaincompress=threshold=0.02:ratio=6:attack=20:release=250[duckedbed]"
                )
                filter_parts.append(
                    "[narr][duckedbed]amix=inputs=2:duration=first:dropout_transition=0,"
                    "alimiter=limit=0.95[aout]"
                )
            else:
                filter_parts.append(
                    "[narr][bed]amix=inputs=2:duration=first:dropout_transition=0,"
                    "alimiter=limit=0.95[aout]"
                )
        else:
            filter_parts.append("[narr]alimiter=limit=0.95[aout]")

        command.extend(
            [
                "-filter_complex",
                ";".join(filter_parts),
                "-map",
                "[aout]",
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ]
        )
        self.ffmpeg_client.run(command)
