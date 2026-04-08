from dataclasses import dataclass, field


@dataclass(slots=True)
class ClipWindow:
    start: float
    end: float
    text: str
    source_segment_indices: list[int] = field(default_factory=list)
