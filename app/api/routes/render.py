import json

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.deps import (
    get_local_media_store_service,
    get_metadata_service,
    get_render_service,
)
from app.core.concurrency import run_blocking
from app.schemas.analysis import ClipCandidate, ClipScoreBreakdown
from app.schemas.subtitles import SubtitlePreferences
from app.schemas.transcription import TranscriptResult
from app.schemas.workflow import RenderedClipResult
from app.services.local_media_store_service import LocalMediaStoreService
from app.services.metadata_service import MetadataService
from app.services.render_service import RenderService

router = APIRouter()


@router.post("/render-clip-upload", response_model=RenderedClipResult)
async def render_clip_upload(
    job_id: str = Form(...),
    project_id: str | None = Form(default=None),
    project_title: str | None = Form(default=None),
    clip_start: float = Form(...),
    clip_end: float = Form(...),
    title_hint: str | None = Form(default=None),
    score: float = Form(default=0.0),
    transcript_json: str = Form(...),
    font_family: str = Form(default="Montserrat ExtraBold"),
    font_size: int = Form(default=64),
    fill_color: str = Form(default="#FFFFFF"),
    stroke_color: str = Form(default="#000000"),
    highlight_color: str = Form(default="#FFD54A"),
    position: str = Form(default="bottom_center"),
    max_chars_per_line: int = Form(default=28),
    max_lines: int = Form(default=2),
    file: UploadFile = File(...),
    render_service: RenderService = Depends(get_render_service),
    metadata_service: MetadataService = Depends(get_metadata_service),
    media_store: LocalMediaStoreService = Depends(get_local_media_store_service),
) -> RenderedClipResult:
    media_uri = await media_store.save_upload(file)
    try:
        media_metadata = await run_blocking(metadata_service.read, media_uri)
        transcript = TranscriptResult.model_validate(json.loads(transcript_json))
        rendered = await run_blocking(
            render_service.render_clips,
            media_uri=media_uri,
            job_id=job_id,
            project_id=project_id,
            project_title=project_title,
            media_metadata=media_metadata,
            transcript=transcript,
            clips=[
                ClipCandidate(
                    start=clip_start,
                    end=clip_end,
                    title_hint=title_hint,
                    transcript_excerpt="",
                    scores=ClipScoreBreakdown(total_score=score),
                )
            ],
            subtitle_prefs=SubtitlePreferences(
                font_family=font_family,
                font_size=font_size,
                fill_color=fill_color,
                stroke_color=stroke_color,
                highlight_color=highlight_color,
                position=position,
                max_chars_per_line=max_chars_per_line,
                max_lines=max_lines,
            ),
        )

        return rendered[0]
    finally:
        media_store.delete_upload(media_uri)
