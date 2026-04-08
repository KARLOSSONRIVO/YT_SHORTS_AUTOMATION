from dataclasses import dataclass, field


@dataclass(slots=True)
class TranscriptWindow:
    start: float
    end: float
    text: str
    segment_indices: list[int] = field(default_factory=list)
