from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.deps import (
    get_local_media_store_service,
    get_workflow_pipeline,
)
from app.pipelines.workflow_pipeline import WorkflowPipeline
from app.schemas.analysis import AnalyzeRequest
from app.schemas.subtitles import SubtitlePreferences
from app.schemas.workflow import ProcessVideoResponse
from app.services.local_media_store_service import LocalMediaStoreService

router = APIRouter()


@router.post("/process-video-upload", response_model=ProcessVideoResponse)
async def process_video_upload(
    job_id: str = Form(...),
    language: str | None = Form(default=None),
    min_clip_duration: float = Form(default=15.0),
    max_clip_duration: float = Form(default=45.0),
    top_k: int = Form(default=3),
    target_keywords: str = Form(default=""),
    font_family: str = Form(default="DejaVu Sans"),
    font_size: int = Form(default=64),
    fill_color: str = Form(default="#FFFFFF"),
    stroke_color: str = Form(default="#000000"),
    position: str = Form(default="bottom_center"),
    max_chars_per_line: int = Form(default=28),
    max_lines: int = Form(default=2),
    file: UploadFile = File(...),
    pipeline: WorkflowPipeline = Depends(get_workflow_pipeline),
    media_store: LocalMediaStoreService = Depends(get_local_media_store_service),
) -> ProcessVideoResponse:
    media_uri = await media_store.save_upload(file)
    try:
        payload = AnalyzeRequest(
            job_id=job_id,
            media_uri=media_uri,
            language=language,
            min_clip_duration=min_clip_duration,
            max_clip_duration=max_clip_duration,
            top_k=top_k,
            target_keywords=[item.strip() for item in target_keywords.split(",") if item.strip()],
            subtitle_prefs=SubtitlePreferences(
                font_family=font_family,
                font_size=font_size,
                fill_color=fill_color,
                stroke_color=stroke_color,
                position=position,
                max_chars_per_line=max_chars_per_line,
                max_lines=max_lines,
            ),
        )
        return await pipeline.run(payload)
    finally:
        media_store.delete_upload(media_uri)
