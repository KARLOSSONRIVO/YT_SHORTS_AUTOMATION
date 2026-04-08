from fastapi import APIRouter

from app.core.config import get_settings
from app.integrations.ffmpeg_client import FFmpegClient
from app.integrations.ffprobe_client import FFprobeClient
from app.integrations.whisper_client import WhisperClient

router = APIRouter()


@router.get("/health")
async def healthcheck() -> dict:
    settings = get_settings()
    ffmpeg_client = FFmpegClient()
    ffprobe_client = FFprobeClient()
    whisper_client = WhisperClient(model_name=settings.whisper_model_size)
    return {
        "status": "ok",
        "service": "yt-shorts-python-worker",
        "ffmpeg_available": ffmpeg_client.is_available(),
        "ffprobe_available": ffprobe_client.is_available(),
        "whisper_available": whisper_client.is_available(),
        "docs_enabled": settings.enable_docs,
    }
