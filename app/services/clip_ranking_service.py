from app.domain.scores import ScoreWeights
from app.schemas.analysis import ClipCandidate


class ClipRankingService:
    def __init__(self, weights: ScoreWeights | None = None) -> None:
        self.weights = weights or ScoreWeights()

    def rank(
        self,
        candidates: list[ClipCandidate],
        min_duration: float,
        max_duration: float,
        top_k: int,
    ) -> list[ClipCandidate]:
        scored = []
        for candidate in candidates:
            total = (
                self.weights.hook * candidate.scores.hook_score
                + self.weights.keyword * candidate.scores.keyword_score
                + self.weights.intensity * candidate.scores.speech_intensity_score
                + self.weights.boundary * candidate.scores.silence_boundary_score
                + self.weights.duration * candidate.scores.duration_fit_score
            )
            candidate.scores.total_score = round(total, 4)
            scored.append(candidate)

        ranked = sorted(scored, key=lambda item: item.scores.total_score, reverse=True)
        return self._dedupe_overlaps(ranked)[:top_k]

    def _dedupe_overlaps(self, clips: list[ClipCandidate], overlap_threshold: float = 0.6) -> list[ClipCandidate]:
        selected: list[ClipCandidate] = []
        for candidate in clips:
            if not any(self._overlap_ratio(candidate, existing) >= overlap_threshold for existing in selected):
                selected.append(candidate)
        return selected

    def _overlap_ratio(self, left: ClipCandidate, right: ClipCandidate) -> float:
        overlap = max(0.0, min(left.end, right.end) - max(left.start, right.start))
        shorter = min(left.end - left.start, right.end - right.start)
        return overlap / shorter if shorter > 0 else 0.0
