from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.deps import get_local_media_store_service, get_transcription_pipeline
from app.pipelines.transcription_pipeline import TranscriptionPipeline
from app.schemas.transcription import TranscriptionRequest, TranscriptionResponse
from app.services.local_media_store_service import LocalMediaStoreService

router = APIRouter()


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    payload: TranscriptionRequest,
    pipeline: TranscriptionPipeline = Depends(get_transcription_pipeline),
) -> TranscriptionResponse:
    return await pipeline.run(payload)


@router.post("/transcribe-upload", response_model=TranscriptionResponse)
async def transcribe_upload(
    job_id: str = Form(...),
    language: str | None = Form(default=None),
    file: UploadFile = File(...),
    pipeline: TranscriptionPipeline = Depends(get_transcription_pipeline),
    media_store: LocalMediaStoreService = Depends(get_local_media_store_service),
) -> TranscriptionResponse:
    media_uri = await media_store.save_upload(file)
    try:
        payload = TranscriptionRequest(
            job_id=job_id,
            media_uri=media_uri,
            language=language,
        )
        return await pipeline.run(payload)
    finally:
        media_store.delete_upload(media_uri)
