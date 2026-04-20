from pathlib import Path
import random
import wave

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
    def __init__(
        self,
        ffmpeg_client,
        output_dir: str,
        music_service=None,
        llm_service=None,
        enable_background_music: bool = True,
        default_music_volume: float = 0.15,
        enable_audio_ducking: bool = True,
    ) -> None:
        self.ffmpeg_client = ffmpeg_client
        self.output_dir = Path(output_dir)
        self.music_service = music_service
        self.llm_service = llm_service
        self.enable_background_music = enable_background_music
        self.default_music_volume = default_music_volume
        self.enable_audio_ducking = enable_audio_ducking

    def render_story_video(self, payload: StoryRenderRequest) -> StoryRenderResponse:
        if not self.ffmpeg_client.is_available():
            raise IntegrationError("ffmpeg is required to render faceless story videos.")
        if not payload.image_paths:
            raise ValidationError("At least one scene image is required for rendering.")

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
