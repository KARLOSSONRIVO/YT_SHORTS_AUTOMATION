from typing import List

from pydantic import BaseModel, Field

from app.schemas.analysis import AnalyzeResponse


class RenderedClipResult(BaseModel):
    clip_index: int
    start: float
    end: float
    title_hint: str | None = None
    score: float
    video_path: str
    video_url: str
    subtitles_path: str


class ProcessVideoResponse(BaseModel):
    job_id: str
    analysis: AnalyzeResponse
    rendered_clips: List[RenderedClipResult] = Field(default_factory=list)
