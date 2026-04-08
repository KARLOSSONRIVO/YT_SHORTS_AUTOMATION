from typing import List, Optional

from pydantic import BaseModel, Field


class MediaStream(BaseModel):
    index: int
    codec_type: str
    codec_name: Optional[str] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None


class MediaMetadata(BaseModel):
    media_uri: str
    duration_seconds: float
    format_name: Optional[str] = None
    size_bytes: Optional[int] = None
    has_audio: bool = False
    has_video: bool = False
    streams: List[MediaStream] = Field(default_factory=list)
