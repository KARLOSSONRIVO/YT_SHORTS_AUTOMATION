import json
from dataclasses import dataclass
from pathlib import Path
import re
import wave

from app.core.exceptions import IntegrationError
from app.integrations.huggingface_client import HuggingFaceClient
from app.integrations.whisper_client import WhisperClient
from app.schemas.faceless_video import (
    StorySubtitleGenerationRequest,
    StorySubtitleGenerationResponse,
    SubtitleCue,
)
from app.utils.output_paths import output_url, stage_output_dir


@dataclass(slots=True)
class TimedWord:
    start: float
    end: float
    text: str


@dataclass(slots=True)
class TimedCue:
    index: int
    start: float
    end: float
    text: str
    words: list[TimedWord]


class FacelessSubtitleService:
    FONT_DIR = Path(__file__).resolve().parents[1] / "fonts" / "__pycache__"
    ACTIVE_WORD_COLOR = "#FFD54A"
    ACTIVE_WORD_SCALE_PERCENT = 118
    ACTIVE_WORD_EXTRA_BORDER = 6
    WORD_FADE_IN_MS = 45
    WORD_FADE_OUT_MS = 70
    WORD_HIGHLIGHT_LAG_SECONDS = 0.18
    WORD_HIGHLIGHT_MAX_LAG_RATIO = 0.25

    def __init__(
        self,
        *,
        output_dir: str,
        huggingface_client: HuggingFaceClient,
        whisper_client: WhisperClient,
        whisper_model: str,
        allow_placeholder_generation: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.huggingface_client = huggingface_client
        self.whisper_client = whisper_client
        self.whisper_model = whisper_model
        self.allow_placeholder_generation = allow_placeholder_generation

    def generate_subtitles(
        self, payload: StorySubtitleGenerationRequest
    ) -> StorySubtitleGenerationResponse:
        stage_dir = stage_output_dir(
            output_dir=self.output_dir,
            project_title=payload.project_title,
            project_id=payload.project_id,
            stage_name="subtitles",
        )
        stage_dir.mkdir(parents=True, exist_ok=True)
        timed_cues = self._build_timed_cues(payload)
        cues = [SubtitleCue(index=cue.index, start=cue.start, end=cue.end, text=cue.text) for cue in timed_cues]

        srt_path = stage_dir / "subtitles.srt"
        ass_path = stage_dir / "subtitles.ass"
        timestamp_json_path = stage_dir / "subtitles.json"

        srt_path.write_text(self._to_srt(cues), encoding="utf-8")
        ass_path.write_text(self._to_ass(timed_cues), encoding="utf-8")
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
            srt_url=output_url(output_dir=self.output_dir, file_path=srt_path),
            ass_url=output_url(output_dir=self.output_dir, file_path=ass_path),
            timestamp_json_url=output_url(output_dir=self.output_dir, file_path=timestamp_json_path),
            subtitles=cues,
        )

    def _build_timed_cues(self, payload: StorySubtitleGenerationRequest) -> list[TimedCue]:
        if not payload.audio_path:
            if self.allow_placeholder_generation:
                return self._scene_timed_cues(payload)
            raise IntegrationError("audio_path is required for Hugging Face Whisper subtitle generation.")

        if self.whisper_client.is_available():
            try:
                cues = self._build_local_whisper_cues(payload)
                if cues:
                    return cues
            except Exception as exc:
                if not self.allow_placeholder_generation and not isinstance(exc, IntegrationError):
                    raise IntegrationError(f"Local Whisper transcription failed: {exc}") from exc
                if isinstance(exc, IntegrationError) and not self.allow_placeholder_generation:
                    raise

        try:
            response = self.huggingface_client.transcribe_audio(
                model=self.whisper_model,
                audio_path=payload.audio_path,
            )
        except Exception as exc:
            if self.allow_placeholder_generation:
                return self._scene_timed_cues(payload)
            if isinstance(exc, IntegrationError):
                raise
            raise IntegrationError(f"Hugging Face Whisper transcription failed: {exc}") from exc

        chunks = response.get("chunks")
        if isinstance(chunks, list) and chunks:
            cues = self._chunks_to_timed_cues(chunks)
            if cues:
                return self._normalize_timed_cues_to_audio(cues=cues, payload=payload)

        text = str(response.get("text") or "").strip()
        if not text:
            raise IntegrationError("Hugging Face Whisper returned an empty transcription.")

        return self._text_to_scene_timed_cues(text=text, payload=payload)

    def _build_local_whisper_cues(self, payload: StorySubtitleGenerationRequest) -> list[TimedCue]:
        transcript = self.whisper_client.transcribe(payload.audio_path, language=None)
        segments = transcript.get("segments", [])
        cues: list[TimedCue] = []
        for index, segment in enumerate(segments, start=1):
            text = str(segment.get("text") or "").strip()
            start = segment.get("start")
            end = segment.get("end")
            if not text or start is None or end is None:
                continue

            words = [
                TimedWord(
                    start=round(float(word.get("start", start)), 2),
                    end=round(float(word.get("end", end)), 2),
                    text=str(word.get("word") or "").strip(),
                )
                for word in (segment.get("words") or [])
                if str(word.get("word") or "").strip()
            ]
            cues.append(
                TimedCue(
                    index=index,
                    start=round(float(start), 2),
                    end=round(float(end), 2),
                    text=text,
                    words=words,
                )
            )

        return self._normalize_timed_cues_to_audio(cues=cues, payload=payload)

    def _scene_timed_cues(self, payload: StorySubtitleGenerationRequest) -> list[TimedCue]:
        cues: list[TimedCue] = []
        cursor = 0.0
        for index, scene in enumerate(payload.scenes, start=1):
            duration = max(scene.duration_seconds, 1.0)
            cues.append(
                TimedCue(
                    index=index,
                    start=round(cursor, 2),
                    end=round(cursor + duration, 2),
                    text=scene.caption_text or scene.narration,
                    words=[],
                )
            )
            cursor += duration
        return cues

    def _normalize_timed_cues_to_audio(
        self,
        *,
        cues: list[TimedCue],
        payload: StorySubtitleGenerationRequest,
    ) -> list[TimedCue]:
        if not payload.audio_path or not cues:
            return cues

        audio_duration = self._audio_duration(Path(payload.audio_path))
        cue_duration = max(cue.end for cue in cues)
        if not audio_duration or cue_duration <= 0 or cue_duration <= audio_duration * 1.05:
            return cues

        scale = audio_duration / cue_duration
        normalized: list[TimedCue] = []
        for cue in cues:
            normalized_words = [
                TimedWord(
                    start=round(word.start * scale, 2),
                    end=round(max(word.end * scale, word.start * scale + 0.05), 2),
                    text=word.text,
                )
                for word in cue.words
            ]
            normalized.append(
                TimedCue(
                    index=cue.index,
                    start=round(cue.start * scale, 2),
                    end=round(max(cue.end * scale, cue.start * scale + 0.1), 2),
                    text=cue.text,
                    words=normalized_words,
                )
            )
        return normalized

    def _chunks_to_timed_cues(self, chunks: list) -> list[TimedCue]:
        cues: list[TimedCue] = []
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
                TimedCue(
                    index=index,
                    start=round(float(start), 2),
                    end=round(float(end), 2),
                    text=text,
                    words=[],
                )
            )
        return cues

    def _text_to_scene_timed_cues(
        self,
        *,
        text: str,
        payload: StorySubtitleGenerationRequest,
    ) -> list[TimedCue]:
        words = text.split()
        if not words:
            return self._scene_timed_cues(payload)

        scene_count = max(len(payload.scenes), 1)
        words_per_scene = max(round(len(words) / scene_count), 1)
        cues: list[TimedCue] = []
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
                TimedCue(
                    index=index,
                    start=round(cursor, 2),
                    end=round(cursor + duration, 2),
                    text=cue_text,
                    words=[],
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

    def _to_ass(self, cues: list[TimedCue]) -> str:
        lines = [
            "[Script Info]",
            "ScriptType: v4.00+",
            "PlayResX: 1080",
            "PlayResY: 1920",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            self._ass_style_line(),
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]
        for cue in cues:
            lines.extend(self._cue_to_ass_events(cue))
        return "\n".join(lines)

    def _ass_style_line(self) -> str:
        return (
            "Style: Default,"
            "Bebas Neue,"
            "156,"
            "&H00FFFFFF,"
            "&H00FFFFFF,"
            "&H00000000,"
            "&H64000000,"
            "1,0,0,0,100,100,0,0,1,5,0,"
            "5,60,60,220,1"
        )

    def _cue_to_ass_events(self, cue: TimedCue) -> list[str]:
        if cue.words:
            return self._word_timed_events(cue)

        tokens = cue.text.split()
        if not tokens:
            return []

        if len(tokens) == 1:
            return [self._single_word_event(tokens[0], cue.start, cue.end)]

        timings = self._word_timings(cue, tokens)
        return [self._single_word_event(tokens[index], start, end) for index, (start, end) in enumerate(timings)]

    def _word_timed_events(self, cue: TimedCue) -> list[str]:
        words = [word for word in cue.words if word.text]
        if not words:
            return self._cue_to_ass_events(
                TimedCue(index=cue.index, start=cue.start, end=cue.end, text=cue.text, words=[])
            )

        events: list[str] = []
        for active_index, word in enumerate(words):
            start = max(word.start, cue.start)
            end = cue.end if active_index == len(words) - 1 else max(words[active_index + 1].start, word.end)
            end = max(end, start + 0.05)
            events.append(self._single_word_event(word.text, start, end))
        return events

    def _word_timings(self, cue: SubtitleCue, tokens: list[str]) -> list[tuple[float, float]]:
        duration = max(cue.end - cue.start, 0.1)
        weights = [max(len(re.sub(r"[^A-Za-z0-9']+", "", token)), 1) for token in tokens]
        total_weight = sum(weights) or len(tokens)

        base_timings: list[tuple[float, float]] = []
        cursor = cue.start
        for index, weight in enumerate(weights):
            if index == len(weights) - 1:
                next_cursor = cue.end
            else:
                next_cursor = cursor + (duration * (weight / total_weight))
            base_timings.append((cursor, max(next_cursor, cursor + 0.05)))
            cursor = next_cursor

        average_word_duration = duration / max(len(tokens), 1)
        lag = min(
            self.WORD_HIGHLIGHT_LAG_SECONDS,
            average_word_duration * self.WORD_HIGHLIGHT_MAX_LAG_RATIO,
        )

        timings: list[tuple[float, float]] = []
        for index, (start, end) in enumerate(base_timings):
            shifted_start = start if index == 0 else min(start + lag, cue.end - 0.05)
            if index + 1 < len(base_timings):
                next_start = base_timings[index + 1][0]
                shifted_end = min(next_start + lag, cue.end)
            else:
                shifted_end = cue.end
            shifted_end = max(shifted_end, shifted_start + 0.05)
            timings.append((round(shifted_start, 2), round(shifted_end, 2)))

        if timings:
            start, _end = timings[-1]
            timings[-1] = (start, round(cue.end, 2))
        return timings

    def _single_word_event(self, token: str, start: float, end: float) -> str:
        escaped = self._escape_ass_text(token.upper())
        style_tag = (
            "{"
            f"\\c{self._hex_to_ass_color(self.ACTIVE_WORD_COLOR)}"
            "\\b1"
            f"\\fscx{self.ACTIVE_WORD_SCALE_PERCENT}"
            f"\\fscy{self.ACTIVE_WORD_SCALE_PERCENT}"
            f"\\bord{self.ACTIVE_WORD_EXTRA_BORDER}"
            f"\\fad({self.WORD_FADE_IN_MS},{self.WORD_FADE_OUT_MS})"
            "}"
        )
        return (
            "Dialogue: 0,"
            f"{self._format_ass_timestamp(start)},"
            f"{self._format_ass_timestamp(end)},"
            f"Default,,0,0,0,,{style_tag}{escaped}{{\\rDefault}}"
        )

    def _escape_ass_text(self, text: str) -> str:
        escaped = text.replace("\\", r"\\")
        escaped = escaped.replace("{", r"\{")
        escaped = escaped.replace("}", r"\}")
        return escaped

    def _hex_to_ass_color(self, value: str) -> str:
        cleaned = value.strip().lstrip("#")
        if len(cleaned) != 6:
            return "&H00FFFFFF"
        rr, gg, bb = cleaned[0:2], cleaned[2:4], cleaned[4:6]
        return f"&H00{bb}{gg}{rr}"

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
