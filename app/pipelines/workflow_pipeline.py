from app.core.concurrency import run_blocking
from app.schemas.analysis import AnalyzeRequest
from app.schemas.workflow import ProcessVideoResponse


class WorkflowPipeline:
    def __init__(self, highlight_pipeline, render_service) -> None:
        self.highlight_pipeline = highlight_pipeline
        self.render_service = render_service

    async def run(self, req: AnalyzeRequest) -> ProcessVideoResponse:
        analysis = await self.highlight_pipeline.run(req)
        rendered_clips = await run_blocking(
            self.render_service.render_clips,
            media_uri=req.media_uri,
            job_id=req.job_id,
            project_id=None,
            project_title=None,
            media_metadata=analysis.media,
            transcript=analysis.transcript,
            clips=analysis.clips,
            subtitle_prefs=req.subtitle_prefs,
        )
        return ProcessVideoResponse(
            job_id=req.job_id,
            analysis=analysis,
            rendered_clips=rendered_clips,
        )
