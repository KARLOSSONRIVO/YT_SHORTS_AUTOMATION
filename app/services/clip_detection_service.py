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
            start = self._snap_boundary(window.start, boundaries, direction="backward")
            end = self._snap_boundary(window.end, boundaries, direction="forward")
            duration = end - start
            if duration < min_duration or duration > max_duration:
                continue

            key = (round(start, 2), round(end, 2))
            if key in seen_ranges:
                continue
            seen_ranges.add(key)

            boundary_score = 1.0 if start != window.start or end != window.end else 0.5
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

    def _snap_boundary(self, value: float, boundaries: list[float], direction: str) -> float:
        if not boundaries:
            return value
        if direction == "backward":
            prior = [item for item in boundaries if item <= value]
            return prior[-1] if prior else value
        later = [item for item in boundaries if item >= value]
        return later[0] if later else value
