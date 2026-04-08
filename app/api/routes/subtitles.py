from fastapi import APIRouter, Depends

from app.api.deps import get_subtitle_pipeline
from app.pipelines.subtitle_pipeline import SubtitlePipeline
from app.schemas.subtitles import SubtitleGenerationRequest, SubtitleGenerationResponse

router = APIRouter()


@router.post("/generate-subtitles", response_model=SubtitleGenerationResponse)
async def generate_subtitles(
    payload: SubtitleGenerationRequest,
    pipeline: SubtitlePipeline = Depends(get_subtitle_pipeline),
) -> SubtitleGenerationResponse:
    return await pipeline.run(payload)
