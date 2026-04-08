import logging

from app.core.exceptions import MediaError
from app.core.telemetry import stage_timer
from app.schemas.transcription import TranscriptionRequest, TranscriptionResponse

logger = logging.getLogger(__name__)


class TranscriptionPipeline:
    def __init__(self, metadata_service, media_prep_service, transcription_service) -> None:
        self.metadata_service = metadata_service
        self.media_prep_service = media_prep_service
        self.transcription_service = transcription_service

    async def run(self, req: TranscriptionRequest) -> TranscriptionResponse:
        logger.info("transcription_pipeline_started", extra={"job_id": req.job_id, "stage": "transcription"})
        with stage_timer("metadata", job_id=req.job_id):
            metadata = self.metadata_service.read(req.media_uri)
            if not metadata.has_audio:
                raise MediaError("Media has no audio stream for transcription.")
        with stage_timer("media_prep", job_id=req.job_id):
            prepared_media_uri = self.media_prep_service.prepare(req.media_uri)
        with stage_timer("transcription", job_id=req.job_id):
            transcript = await self.transcription_service.transcribe(prepared_media_uri, req.language)
        return TranscriptionResponse(job_id=req.job_id, media=metadata, transcript=transcript)
