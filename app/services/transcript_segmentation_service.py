from app.domain.transcript import TranscriptWindow
from app.schemas.transcription import TranscriptResult
from app.utils.text import compact_whitespace


class TranscriptSegmentationService:
    def build_windows(
        self,
        transcript: TranscriptResult,
        min_duration: float,
        max_duration: float,
    ) -> list[TranscriptWindow]:
        segments = transcript.segments
        if not segments:
            return []

        windows: list[TranscriptWindow] = []
        total_segments = len(segments)

        for start_idx in range(total_segments):
            start = segments[start_idx].start
            text_parts: list[str] = []
            indices: list[int] = []

            for end_idx in range(start_idx, total_segments):
                current = segments[end_idx]
                text_parts.append(current.text)
                indices.append(end_idx)
                duration = current.end - start

                if duration < min_duration:
                    continue
                if duration > max_duration:
                    break

                windows.append(
                    TranscriptWindow(
                        start=start,
                        end=current.end,
                        text=compact_whitespace(" ".join(text_parts)),
                        segment_indices=indices.copy(),
                    )
                )

        return windows
