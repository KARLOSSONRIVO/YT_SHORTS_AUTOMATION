from app.schemas.subtitles import SubtitleGenerationRequest, SubtitleGenerationResponse


class SubtitlePipeline:
    def __init__(self, subtitle_formatting_service) -> None:
        self.subtitle_formatting_service = subtitle_formatting_service

    async def run(self, req: SubtitleGenerationRequest) -> SubtitleGenerationResponse:
        subtitles = self.subtitle_formatting_service.build_from_transcript(req)
        return SubtitleGenerationResponse(job_id=req.job_id, subtitles=subtitles)
