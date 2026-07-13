import json
from dataclasses import dataclass
from pathlib import Path
import re
import wave

from app.core.exceptions import IntegrationError
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
    whole_line: bool = False


class FacelessSubtitleService:
    FONT_DIR = Path(__file__).resolve().parents[1] / "fonts" / "__pycache__"
    ACTIVE_WORD_COLOR = "#FFD54A"
    ACTIVE_WORD_SCALE_PERCENT = 118
    ACTIVE_WORD_EXTRA_BORDER = 6
    WORD_FADE_IN_MS = 45
    WORD_FADE_OUT_MS = 70
    WORD_HIGHLIGHT_LAG_SECONDS = 0.18
    WORD_HIGHLIGHT_MAX_LAG_RATIO = 0.25
    TITLE_INTRO_HOLD_SECONDS = 0.35
    OPENING_TEXT_DELAY_SECONDS = 0.16
    OPENING_TEXT_FADE_IN_MS = 180
    OPENING_TEXT_FADE_OUT_MS = 110
    OPENING_TEXT_MOVE_DISTANCE_PX = 42

    def __init__(
        self,
        *,
        output_dir: str,
        whisper_client: WhisperClient,
        allow_placeholder_generation: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.whisper_client = whisper_client
        self.allow_placeholder_generation = allow_placeholder_generation

    def generate_subtitles(
        self, payload: StorySubtitleGenerationRequest
    ) -> StorySubtitleGenerationResponse:
        stage_dir = stage_output_dir(
            output_dir=self.output_dir,
            output_bucket=payload.output_bucket,
            project_title=payload.project_title,
            project_id=payload.project_id,
            stage_name="subtitles",
        )
        stage_dir.mkdir(parents=True, exist_ok=True)
        timed_cues = self._collapse_opening_full_line(self._build_timed_cues(payload), payload)
        cues = [
            SubtitleCue(index=cue.index, start=cue.start, end=cue.end, text=self._clean_display_text(cue.text))
            for cue in timed_cues
        ]

        srt_path = stage_dir / "subtitles.srt"
        ass_path = stage_dir / "subtitles.ass"
        timestamp_json_path = stage_dir / "subtitles.json"

        srt_path.write_text(self._to_srt(cues), encoding="utf-8")
        ass_path.write_text(self._to_ass(timed_cues, payload), encoding="utf-8")
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
            raise IntegrationError("audio_path is required for faster-whisper subtitle generation.")

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

        if self.allow_placeholder_generation:
            return self._scene_timed_cues(payload)
        raise IntegrationError("faster-whisper is required for faceless subtitle alignment.")

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
                    whole_line=False,
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
                    whole_line=False,
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
                    whole_line=cue.whole_line,
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
                    whole_line=False,
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
                    whole_line=False,
                )
            )
            cursor += duration

        return cues

    def _collapse_opening_full_line(
        self,
        cues: list[TimedCue],
        payload: StorySubtitleGenerationRequest,
    ) -> list[TimedCue]:
        if not cues:
            return cues

        opening_text = (payload.opening_display_text or payload.project_title or "").strip()
        if not opening_text:
            return cues

        opening_audio_end = self._opening_audio_end(cues=cues, opening_text=opening_text)
        if opening_audio_end is None:
            first_cue_end = cues[0].end if cues else self._estimate_opening_duration(opening_text)
            opening_audio_end = min(first_cue_end, self._estimate_opening_duration(opening_text))

        actual_opening_end = round(opening_audio_end + self.TITLE_INTRO_HOLD_SECONDS, 2)
        collapsed_opening_cue = TimedCue(
            index=1,
            start=round(self.OPENING_TEXT_DELAY_SECONDS, 2),
            end=actual_opening_end,
            text=opening_text,
            words=[],
            whole_line=True,
        )

        remaining: list[TimedCue] = []
        for cue in cues:
            if cue.end <= actual_opening_end:
                continue

            if cue.start < actual_opening_end:
                clipped_words = [
                    TimedWord(
                        start=round(max(word.start, actual_opening_end), 2),
                        end=word.end,
                        text=word.text,
                    )
                    for word in cue.words
                    if word.end > actual_opening_end
                ]
                cue = TimedCue(
                    index=cue.index,
                    start=actual_opening_end,
                    end=cue.end,
                    text=" ".join(word.text for word in clipped_words).strip() or cue.text,
                    words=clipped_words,
                    whole_line=cue.whole_line,
                )

            remaining.append(cue)

        normalized_cues = [collapsed_opening_cue]
        for index, cue in enumerate(remaining, start=2):
            normalized_cues.append(
                TimedCue(
                    index=index,
                    start=cue.start,
                    end=cue.end,
                    text=cue.text,
                    words=cue.words,
                    whole_line=cue.whole_line,
                )
            )
        return normalized_cues

    def _opening_audio_end(self, *, cues: list[TimedCue], opening_text: str) -> float | None:
        opening_tokens = self._normalize_text(opening_text).split()
        if not opening_tokens:
            return None

        matched_index = 0
        last_match_end: float | None = None
        minimum_partial_match = max(round(len(opening_tokens) * 0.7), 1)

        for cue in cues:
            if cue.start > 8.0:
                break

            if not cue.words:
                normalized_cue = self._normalize_text(cue.text)
                normalized_opening = " ".join(opening_tokens)
                if (
                    normalized_cue == normalized_opening
                    or normalized_cue.startswith(normalized_opening)
                    or normalized_opening.startswith(normalized_cue)
                ):
                    return cue.end
                continue

            for word in cue.words:
                for token in self._normalize_text(word.text).split():
                    if matched_index < len(opening_tokens) and token == opening_tokens[matched_index]:
                        matched_index += 1
                        last_match_end = word.end
                        if matched_index == len(opening_tokens):
                            return last_match_end
                        continue

                    if matched_index >= minimum_partial_match:
                        return last_match_end

                    if matched_index > 0:
                        return None

        if matched_index >= minimum_partial_match:
            return last_match_end
        return None

    def _estimate_opening_duration(self, opening_text: str) -> float:
        word_count = max(len(self._normalize_text(opening_text).split()), 1)
        return min(max(word_count / 2.35, 0.9), 3.5)

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

    def _to_ass(self, cues: list[TimedCue], payload: StorySubtitleGenerationRequest) -> str:
        lines = [
            "[Script Info]",
            "ScriptType: v4.00+",
            "PlayResX: 1080",
            "PlayResY: 1920",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            self._ass_style_line(payload),
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]
        for cue in cues:
            lines.extend(self._cue_to_ass_events(cue, payload))
        return "\n".join(lines)

    def _ass_style_line(self, payload: StorySubtitleGenerationRequest) -> str:
        font_name = payload.font_family or "Montserrat ExtraBold"
        font_size = payload.font_size or 64
        primary_color = self._hex_to_ass_color(payload.fill_color or "#FFFFFF")
        outline_color = self._hex_to_ass_color(payload.stroke_color or "#000000")
        alignment = self._ass_alignment(payload.position)
        margin_v = self._ass_margin_vertical(payload.position)
        return (
            "Style: Default,"
            f"{font_name},"
            f"{font_size},"
            f"{primary_color},"
            f"{primary_color},"
            f"{outline_color},"
            "&H64000000,"
            "1,0,0,0,100,100,0,0,1,5,0,"
            f"{alignment},60,60,{margin_v},1"
        )

    def _cue_to_ass_events(self, cue: TimedCue, payload: StorySubtitleGenerationRequest) -> list[str]:
        if cue.whole_line:
            return [self._full_line_event(cue.text, cue.start, cue.end, payload)]

        if cue.words:
            return self._word_timed_events(cue, payload)

        tokens = self._clean_display_text(cue.text).split()
        if not tokens:
            return []

        if len(tokens) == 1:
            return [self._single_word_event(tokens[0], cue.start, cue.end, payload)]

        timings = self._word_timings(cue, tokens)
        return [
            self._single_word_event(tokens[index], start, end, payload)
            for index, (start, end) in enumerate(timings)
        ]

    def _word_timed_events(self, cue: TimedCue, payload: StorySubtitleGenerationRequest) -> list[str]:
        words = [
            TimedWord(start=word.start, end=word.end, text=self._clean_display_text(word.text))
            for word in cue.words
            if self._clean_display_text(word.text)
        ]
        if not words:
            return self._cue_to_ass_events(
                TimedCue(index=cue.index, start=cue.start, end=cue.end, text=cue.text, words=[]),
                payload
            )

        events: list[str] = []
        for active_index, word in enumerate(words):
            start = max(word.start, cue.start)
            end = cue.end if active_index == len(words) - 1 else max(words[active_index + 1].start, word.end)
            end = max(end, start + 0.05)
            events.append(self._single_word_event(word.text, start, end, payload))
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

    def _single_word_event(
        self,
        token: str,
        start: float,
        end: float,
        payload: StorySubtitleGenerationRequest,
    ) -> str:
        escaped = self._escape_ass_text(token.upper())
        ass_color = self._hex_to_ass_color(payload.highlight_color or self.ACTIVE_WORD_COLOR)
        position_tag = self._ass_position_override(payload.position)
        style_tag = (
            "{"
            f"{position_tag}"
            f"\\1c{ass_color}"
            f"\\c{ass_color}"
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

    def _full_line_event(
        self,
        text: str,
        start: float,
        end: float,
        payload: StorySubtitleGenerationRequest,
    ) -> str:
        escaped = self._escape_ass_text(self._clean_display_text(text))
        position_tag = self._ass_position_override(payload.position, animated=True)
        return (
            "Dialogue: 0,"
            f"{self._format_ass_timestamp(start)},"
            f"{self._format_ass_timestamp(end)},"
            f"Default,,0,0,0,,{{{position_tag}\\fad({self.OPENING_TEXT_FADE_IN_MS},{self.OPENING_TEXT_FADE_OUT_MS})}}{escaped}"
        )

    def _ass_alignment(self, position: str | None) -> int:
        if position == "top_center":
            return 8
        if position == "middle_center":
            return 5
        return 2

    def _ass_margin_vertical(self, position: str | None) -> int:
        if position == "top_center":
            return 180
        if position == "middle_center":
            return 120
        return 220

    def _ass_position_override(self, position: str | None, animated: bool = False) -> str:
        x = self.TARGET_CENTER_X()
        y = self.TARGET_POSITION_Y(position)
        alignment = self._ass_alignment(position)
        if animated:
            return (
                f"\\an{alignment}"
                f"\\move({x},{y + self.OPENING_TEXT_MOVE_DISTANCE_PX},{x},{y},0,{self.OPENING_TEXT_FADE_IN_MS})"
            )
        return f"\\an{alignment}\\pos({x},{y})"

    def TARGET_CENTER_X(self) -> int:
        return 540

    def TARGET_POSITION_Y(self, position: str | None) -> int:
        if position == "top_center":
            return 320
        if position == "middle_center":
            return 960
        return 1560

    def _escape_ass_text(self, text: str) -> str:
        escaped = text.replace("\\", r"\\")
        escaped = escaped.replace("{", r"\{")
        escaped = escaped.replace("}", r"\}")
        return escaped

    def _normalize_text(self, value: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s']+", " ", value.lower())).strip()

    def _clean_display_text(self, value: str) -> str:
        cleaned = re.sub(r"[_\W]+", " ", value, flags=re.UNICODE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned or value.strip()

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
