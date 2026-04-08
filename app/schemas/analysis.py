from typing import List, Optional

from pydantic import BaseModel, Field

from app.schemas.media import MediaMetadata
from app.schemas.subtitles import SubtitlePreferences, SubtitleSegment
from app.schemas.transcription import TranscriptResult


class ClipScoreBreakdown(BaseModel):
    hook_score: float = 0.0
    keyword_score: float = 0.0
    speech_intensity_score: float = 0.0
    silence_boundary_score: float = 0.0
    duration_fit_score: float = 0.0
    total_score: float = 0.0


class ClipCandidate(BaseModel):
    start: float
    end: float
    title_hint: Optional[str] = None
    transcript_excerpt: str
    scores: ClipScoreBreakdown


class AnalyzeRequest(BaseModel):
    job_id: str
    media_uri: str
    language: Optional[str] = None
    min_clip_duration: float = Field(default=15.0, gt=0)
    max_clip_duration: float = Field(default=45.0, gt=0)
    top_k: int = Field(default=5, ge=1, le=20)
    target_keywords: List[str] = Field(default_factory=list)
    subtitle_prefs: SubtitlePreferences = Field(default_factory=SubtitlePreferences)


class AnalyzeResponse(BaseModel):
    job_id: str
    media: MediaMetadata
    transcript: TranscriptResult
    clips: List[ClipCandidate]
    subtitles: List[SubtitleSegment]
    warnings: List[str] = Field(default_factory=list)
