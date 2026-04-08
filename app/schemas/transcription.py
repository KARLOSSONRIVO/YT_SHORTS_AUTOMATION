from typing import List, Optional

from pydantic import BaseModel, Field

from app.schemas.media import MediaMetadata


class TranscriptWord(BaseModel):
    start: float
    end: float
    word: str
    probability: Optional[float] = None


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str
    avg_logprob: Optional[float] = None
    words: List[TranscriptWord] = Field(default_factory=list)


class TranscriptResult(BaseModel):
    language: str
    duration: float
    segments: List[TranscriptSegment]
    full_text: str


class TranscriptionRequest(BaseModel):
    job_id: str
    media_uri: str
    language: Optional[str] = None


class TranscriptionResponse(BaseModel):
    job_id: str
    media: MediaMetadata
    transcript: TranscriptResult
