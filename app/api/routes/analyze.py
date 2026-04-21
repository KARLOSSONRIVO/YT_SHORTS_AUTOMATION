from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.deps import get_highlight_pipeline, get_local_media_store_service
from app.pipelines.highlight_pipeline import HighlightPipeline
from app.schemas.analysis import AnalyzeRequest, AnalyzeResponse
from app.services.local_media_store_service import LocalMediaStoreService

router = APIRouter()


@router.post("/analyze-clips", response_model=AnalyzeResponse)
async def analyze_clips(
    payload: AnalyzeRequest,
    pipeline: HighlightPipeline = Depends(get_highlight_pipeline),
) -> AnalyzeResponse:
    return await pipeline.run(payload)


@router.post("/analyze-clips-upload", response_model=AnalyzeResponse)
async def analyze_clips_upload(
    job_id: str = Form(...),
    language: str | None = Form(default=None),
    min_clip_duration: float = Form(default=15.0),
    max_clip_duration: float = Form(default=45.0),
    top_k: int = Form(default=5),
    target_keywords: str = Form(default=""),
    file: UploadFile = File(...),
    pipeline: HighlightPipeline = Depends(get_highlight_pipeline),
    media_store: LocalMediaStoreService = Depends(get_local_media_store_service),
) -> AnalyzeResponse:
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
        )
        return await pipeline.run(payload)
    finally:
        media_store.delete_upload(media_uri)
