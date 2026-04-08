from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.transcription import TranscriptResult


class SubtitlePreferences(BaseModel):
    font_family: str = "Montserrat ExtraBold"
    font_size: int = 64
    fill_color: str = "#FFFFFF"
    stroke_color: str = "#000000"
    highlight_color: str = "#FFD54A"
    position: str = "bottom_center"
    max_chars_per_line: int = 28
    max_lines: int = 2


class SubtitleSegment(BaseModel):
    start: float
    end: float
    text: str


class SubtitleGenerationRequest(BaseModel):
    job_id: str
    transcript: TranscriptResult
    clip_start: Optional[float] = None
    clip_end: Optional[float] = None
    max_chars_per_line: int = 28
    max_lines: int = 2


class SubtitleGenerationResponse(BaseModel):
    job_id: str
    subtitles: list[SubtitleSegment] = Field(default_factory=list)
