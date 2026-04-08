from app.schemas.transcription import TranscriptResult


class SilenceDetectionService:
    def detect(self, transcript: TranscriptResult, min_gap_seconds: float = 0.45) -> list[float]:
        boundaries: list[float] = []
        segments = transcript.segments
        for idx in range(len(segments) - 1):
            gap = segments[idx + 1].start - segments[idx].end
            if gap >= min_gap_seconds:
                boundaries.append(segments[idx].end)
                boundaries.append(segments[idx + 1].start)
        return sorted(set(boundaries))
