import json
from pathlib import Path
import wave

from app.core.exceptions import IntegrationError
from app.integrations.huggingface_client import HuggingFaceClient
from app.schemas.faceless_video import (
    StorySubtitleGenerationRequest,
    StorySubtitleGenerationResponse,
    SubtitleCue,
)


class FacelessSubtitleService:
    def __init__(
        self,
        *,
        output_dir: str,
        huggingface_client: HuggingFaceClient,
        whisper_model: str,
        allow_placeholder_generation: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.huggingface_client = huggingface_client
        self.whisper_model = whisper_model
        self.allow_placeholder_generation = allow_placeholder_generation

    def generate_subtitles(
        self, payload: StorySubtitleGenerationRequest
    ) -> StorySubtitleGenerationResponse:
        job_dir = self.output_dir / payload.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        cues = self._build_huggingface_cues(payload)

        srt_path = job_dir / "subtitles.srt"
        ass_path = job_dir / "subtitles.ass"
        timestamp_json_path = job_dir / "subtitles.json"

        srt_path.write_text(self._to_srt(cues), encoding="utf-8")
        ass_path.write_text(self._to_ass(cues), encoding="utf-8")
        timestamp_json_path.write_text(
            json.dumps([cue.model_dump() for cue in cues], indent=2),
            encoding="utf-8",
        )

        return StorySubtitleGenerationResponse(
            job_id=payload.job_id,
            project_id=payload.project_id,
            srt_path=str(srt_path.resolve()),
            ass_path=str(ass_path.resolve()),
            timestamp_json_path=str(timestamp_json_path.resolve()),
            srt_url=f"/outputs/{payload.job_id}/{srt_path.name}",
            ass_url=f"/outputs/{payload.job_id}/{ass_path.name}",
            timestamp_json_url=f"/outputs/{payload.job_id}/{timestamp_json_path.name}",
            subtitles=cues,
        )

    def _build_huggingface_cues(self, payload: StorySubtitleGenerationRequest) -> list[SubtitleCue]:
        if not payload.audio_path:
            if self.allow_placeholder_generation:
                return self._build_scene_cues(payload)
            raise IntegrationError("audio_path is required for Hugging Face Whisper subtitle generation.")

        try:
            response = self.huggingface_client.transcribe_audio(
                model=self.whisper_model,
                audio_path=payload.audio_path,
            )
        except Exception as exc:
            if self.allow_placeholder_generation:
                return self._build_scene_cues(payload)
            if isinstance(exc, IntegrationError):
                raise
            raise IntegrationError(f"Hugging Face Whisper transcription failed: {exc}") from exc

        chunks = response.get("chunks")
        if isinstance(chunks, list) and chunks:
            cues = self._chunks_to_cues(chunks)
            if cues:
                return self._normalize_cues_to_audio(cues=cues, payload=payload)

        text = str(response.get("text") or "").strip()
        if not text:
            raise IntegrationError("Hugging Face Whisper returned an empty transcription.")

        return self._text_to_scene_timed_cues(text=text, payload=payload)

    def _build_cues(self, payload: StorySubtitleGenerationRequest) -> list[SubtitleCue]:
        return self._build_scene_cues(payload)

    def _build_scene_cues(self, payload: StorySubtitleGenerationRequest) -> list[SubtitleCue]:
        cues: list[SubtitleCue] = []
        cursor = 0.0
        for index, scene in enumerate(payload.scenes, start=1):
            duration = max(scene.duration_seconds, 1.0)
            cues.append(
                SubtitleCue(
                    index=index,
                    start=round(cursor, 2),
                    end=round(cursor + duration, 2),
                    text=scene.caption_text or scene.narration,
                )
            )
            cursor += duration
        return cues

    def _normalize_cues_to_audio(
        self,
        *,
        cues: list[SubtitleCue],
        payload: StorySubtitleGenerationRequest,
    ) -> list[SubtitleCue]:
        if not payload.audio_path or not cues:
            return cues

        audio_duration = self._audio_duration(Path(payload.audio_path))
        cue_duration = max(cue.end for cue in cues)
        if not audio_duration or cue_duration <= 0 or cue_duration <= audio_duration * 1.05:
            return cues

        scale = audio_duration / cue_duration
        normalized = [
            SubtitleCue(
                index=cue.index,
                start=round(cue.start * scale, 2),
                end=round(max(cue.end * scale, cue.start * scale + 0.1), 2),
                text=cue.text,
            )
            for cue in cues
        ]
        return normalized

    def _chunks_to_cues(self, chunks: list) -> list[SubtitleCue]:
        cues: list[SubtitleCue] = []
        for index, chunk in enumerate(chunks, start=1):
            if not isinstance(chunk, dict):
                continue

            timestamp = chunk.get("timestamp")
            if not isinstance(timestamp, list | tuple) or len(timestamp) != 2:
                continue

            start, end = timestamp
            text = str(chunk.get("text") or "").strip()
            if not text or start is None or end is None:
                continue

            cues.append(
                SubtitleCue(
                    index=index,
                    start=round(float(start), 2),
                    end=round(float(end), 2),
                    text=text,
                )
            )
        return cues

    def _text_to_scene_timed_cues(
        self,
        *,
        text: str,
        payload: StorySubtitleGenerationRequest,
    ) -> list[SubtitleCue]:
        words = text.split()
        if not words:
            return self._build_scene_cues(payload)

        scene_count = max(len(payload.scenes), 1)
        words_per_scene = max(round(len(words) / scene_count), 1)
        cues: list[SubtitleCue] = []
        cursor = 0.0
        audio_duration = self._audio_duration(Path(payload.audio_path)) if payload.audio_path else None
        scene_total_duration = sum(max(scene.duration_seconds, 1.0) for scene in payload.scenes)
        scale = audio_duration / scene_total_duration if audio_duration and scene_total_duration > 0 else 1.0

        for index, scene in enumerate(payload.scenes, start=1):
            start_word = (index - 1) * words_per_scene
            end_word = len(words) if index == scene_count else index * words_per_scene
            cue_text = " ".join(words[start_word:end_word]).strip() or scene.caption_text
            duration = max(scene.duration_seconds * scale, 1.0)
            cues.append(
                SubtitleCue(
                    index=index,
                    start=round(cursor, 2),
                    end=round(cursor + duration, 2),
                    text=cue_text,
                )
            )
            cursor += duration

        return cues

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

    def _to_srt(self, cues: list[SubtitleCue]) -> str:
        blocks = []
        for cue in cues:
            blocks.append(
                f"{cue.index}\n"
                f"{self._format_srt_timestamp(cue.start)} --> {self._format_srt_timestamp(cue.end)}\n"
                f"{cue.text}"
            )
        return "\n\n".join(blocks)

    def _to_ass(self, cues: list[SubtitleCue]) -> str:
        lines = [
            "[Script Info]",
            "ScriptType: v4.00+",
            "PlayResX: 1080",
            "PlayResY: 1920",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            "Style: Default,Montserrat ExtraBold,46,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,1,0,0,0,100,100,0,0,1,3,0,2,80,80,220,1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]
        for cue in cues:
            text = cue.text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")
            lines.append(
                f"Dialogue: 0,{self._format_ass_timestamp(cue.start)},"
                f"{self._format_ass_timestamp(cue.end)},Default,,0,0,0,,{text}"
            )
        return "\n".join(lines)

    def _format_srt_timestamp(self, seconds: float) -> str:
        total_ms = max(round(seconds * 1000), 0)
        hours = total_ms // 3_600_000
        minutes = (total_ms % 3_600_000) // 60_000
        secs = (total_ms % 60_000) // 1000
        milliseconds = total_ms % 1000
        return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

    def _format_ass_timestamp(self, seconds: float) -> str:
        total_cs = max(round(seconds * 100), 0)
        hours = total_cs // 360000
        minutes = (total_cs % 360000) // 6000
        secs = (total_cs % 6000) // 100
        centiseconds = total_cs % 100
        return f"{hours}:{minutes:02}:{secs:02}.{centiseconds:02}"
