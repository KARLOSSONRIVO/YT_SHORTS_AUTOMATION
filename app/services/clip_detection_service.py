from app.schemas.analysis import ClipCandidate, ClipScoreBreakdown
from app.schemas.transcription import TranscriptResult
from app.utils.text import summarize_excerpt


class ClipDetectionService:
    def generate_candidates(
        self,
        transcript: TranscriptResult,
        windows,
        boundaries: list[float],
        min_duration: float,
        max_duration: float,
        target_keywords: list[str],
        top_k: int,
    ) -> list[ClipCandidate]:
        candidates: list[ClipCandidate] = []
        seen_ranges: set[tuple[float, float]] = set()
        for window in windows:
            if not self._has_natural_boundary(transcript, window, side="start"):
                continue
            if not self._has_natural_boundary(transcript, window, side="end"):
                continue

            start = window.start
            end = window.end
            duration = end - start
            if duration < min_duration or duration > max_duration:
                continue

            key = (round(start, 2), round(end, 2))
            if key in seen_ranges:
                continue
            seen_ranges.add(key)

            boundary_score = self._boundary_score(transcript, window)
            duration_midpoint = (min_duration + max_duration) / 2
            duration_fit = 1.0 - min(abs(duration - duration_midpoint) / max(duration_midpoint, 1), 1.0)

            candidates.append(
                ClipCandidate(
                    start=round(start, 2),
                    end=round(end, 2),
                    title_hint=summarize_excerpt(window.text),
                    transcript_excerpt=window.text,
                    scores=ClipScoreBreakdown(
                        silence_boundary_score=boundary_score,
                        duration_fit_score=duration_fit,
                    ),
                )
            )

        return candidates

    def _has_natural_boundary(self, transcript: TranscriptResult, window, side: str) -> bool:
        if not window.segment_indices:
            return False

        segments = transcript.segments
        if side == "start":
            index = window.segment_indices[0]
            if index == 0:
                return True
            previous = segments[index - 1]
            current = segments[index]
            return current.start - previous.end >= 0.35 or previous.text.rstrip().endswith((".", "!", "?"))

        index = window.segment_indices[-1]
        if index == len(segments) - 1:
            return True
        current = segments[index]
        following = segments[index + 1]
        return following.start - current.end >= 0.35 or current.text.rstrip().endswith((".", "!", "?"))

    def _boundary_score(self, transcript: TranscriptResult, window) -> float:
        if not window.segment_indices:
            return 0.0

        segments = transcript.segments
        first_index = window.segment_indices[0]
        last_index = window.segment_indices[-1]
        first_segment = segments[first_index]
        last_segment = segments[last_index]

        starts_naturally = (
            first_index == 0
            or first_segment.start - segments[first_index - 1].end >= 0.35
            or segments[first_index - 1].text.rstrip().endswith((".", "!", "?"))
        )
        ends_naturally = (
            last_index == len(segments) - 1
            or segments[last_index + 1].start - last_segment.end >= 0.35
            or last_segment.text.rstrip().endswith((".", "!", "?"))
        )

        return (float(starts_naturally) + float(ends_naturally)) / 2
