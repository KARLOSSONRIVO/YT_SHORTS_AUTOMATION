from pathlib import Path
import wave

from app.core.exceptions import IntegrationError, ValidationError
from app.schemas.faceless_video import StoryRenderRequest, StoryRenderResponse


class StoryRenderService:
    def __init__(self, ffmpeg_client, output_dir: str) -> None:
        self.ffmpeg_client = ffmpeg_client
        self.output_dir = Path(output_dir)

    def render_story_video(self, payload: StoryRenderRequest) -> StoryRenderResponse:
        if not self.ffmpeg_client.is_available():
            raise IntegrationError("ffmpeg is required to render faceless story videos.")
        if not payload.image_paths:
            raise ValidationError("At least one scene image is required for rendering.")

        job_dir = self.output_dir / payload.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        concat_path = job_dir / "scene_inputs.txt"
        output_path = job_dir / "faceless_story.mp4"
        audio_duration = self._audio_duration(Path(payload.audio_path))
        scene_total_duration = sum(
            max(scene.duration_seconds if scene else 5.0, 1.0)
            for scene in payload.scenes[: len(payload.image_paths)]
        )
        duration_scale = (
            audio_duration / scene_total_duration
            if audio_duration and scene_total_duration > 0
            else 1.0
        )

        concat_lines = []
        total_duration = 0.0
        for index, image_path in enumerate(payload.image_paths):
            scene = payload.scenes[index] if index < len(payload.scenes) else None
            duration = max((scene.duration_seconds if scene else 5.0) * duration_scale, 1.0)
            concat_lines.append(f"file '{Path(image_path).resolve().as_posix()}'")
            concat_lines.append(f"duration {duration}")
            total_duration += duration

        concat_lines.append(f"file '{Path(payload.image_paths[-1]).resolve().as_posix()}'")
        concat_path.write_text("\n".join(concat_lines), encoding="utf-8")

        filters = [
            "scale=1080:1920:force_original_aspect_ratio=increase",
            "crop=1080:1920",
            "format=yuv420p",
        ]
        if payload.subtitles_path:
            filters.append(f"subtitles='{self._escape_filter_path(Path(payload.subtitles_path))}'")

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
            payload.audio_path,
            "-vf",
            ",".join(filters),
            "-t",
            str(max(audio_duration or total_duration, 1.0)),
            "-r",
            "30",
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
            video_url=f"/outputs/{payload.job_id}/{output_path.name}",
            duration_seconds=round(total_duration, 2),
        )

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
