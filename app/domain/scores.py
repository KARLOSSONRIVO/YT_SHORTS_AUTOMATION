from dataclasses import dataclass


@dataclass(slots=True)
class ScoreWeights:
    hook: float = 0.35
    keyword: float = 0.25
    intensity: float = 0.20
    boundary: float = 0.10
    duration: float = 0.10
