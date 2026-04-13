from fastapi import APIRouter, Depends

from app.api.deps import (
    get_faceless_subtitle_service,
    get_image_service,
    get_llm_service,
    get_story_render_service,
    get_tts_service,
)
from app.schemas.faceless_video import (
    AudioGenerationRequest,
    AudioGenerationResponse,
    SceneImageGenerationRequest,
    SceneImageGenerationResponse,
    ScriptGenerationRequest,
    ScriptGenerationResponse,
    StoryRenderRequest,
    StoryRenderResponse,
    StorySubtitleGenerationRequest,
    StorySubtitleGenerationResponse,
)
from app.services.faceless_subtitle_service import FacelessSubtitleService
from app.services.image_service import ImageService
from app.services.llm_service import LLMService
from app.services.story_render_service import StoryRenderService
from app.services.tts_service import TTSService

router = APIRouter(prefix="/faceless")


@router.post("/generate-script", response_model=ScriptGenerationResponse)
async def generate_script(
    payload: ScriptGenerationRequest,
    llm_service: LLMService = Depends(get_llm_service),
) -> ScriptGenerationResponse:
    return llm_service.generate_story_script(payload)


@router.post("/generate-audio", response_model=AudioGenerationResponse)
async def generate_audio(
    payload: AudioGenerationRequest,
    tts_service: TTSService = Depends(get_tts_service),
) -> AudioGenerationResponse:
    return tts_service.generate_narration(payload)


@router.post("/generate-subtitles", response_model=StorySubtitleGenerationResponse)
async def generate_subtitles(
    payload: StorySubtitleGenerationRequest,
    subtitle_service: FacelessSubtitleService = Depends(get_faceless_subtitle_service),
) -> StorySubtitleGenerationResponse:
    return subtitle_service.generate_subtitles(payload)


@router.post("/generate-scenes", response_model=SceneImageGenerationResponse)
async def generate_scenes(
    payload: SceneImageGenerationRequest,
    image_service: ImageService = Depends(get_image_service),
) -> SceneImageGenerationResponse:
    return image_service.generate_scene_images(payload)


@router.post("/render", response_model=StoryRenderResponse)
async def render_story(
    payload: StoryRenderRequest,
    render_service: StoryRenderService = Depends(get_story_render_service),
) -> StoryRenderResponse:
    return render_service.render_story_video(payload)
