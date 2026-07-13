import shutil
import hashlib
from pathlib import Path

from fastapi import APIRouter, Depends

from app.api.deps import (
    get_ai_animation_service,
    get_faceless_subtitle_service,
    get_image_service,
    get_llm_service,
    get_music_service,
    get_story_render_service,
    get_tts_service,
)
from app.schemas.faceless_video import (
    AudioGenerationRequest,
    AudioGenerationResponse,
    SceneAnimationGenerationRequest,
    SceneAnimationGenerationResponse,
    SceneAmbienceGenerationRequest,
    SceneAmbienceGenerationResponse,
    SceneImageGenerationRequest,
    SceneImageGenerationResponse,
    ScriptGenerationRequest,
    ScriptGenerationResponse,
    StoryRenderRequest,
    StoryRenderResponse,
    StoryMusicGenerationRequest,
    StoryMusicGenerationResponse,
    StorySubtitleGenerationRequest,
    StorySubtitleGenerationResponse,
    ProjectOutputCleanupRequest,
    ProjectOutputCleanupResponse,
    VoiceOption,
    VoicePreviewRequest,
    VoicePreviewResponse,
)
from app.services.ai_animation_service import AIAnimationService
from app.services.faceless_subtitle_service import FacelessSubtitleService
from app.services.image_service import ImageService
from app.services.llm_service import LLMService
from app.services.music_service import MusicService
from app.services.story_render_service import StoryRenderService
from app.services.tts_service import TTSService
from app.utils.output_paths import iter_project_output_roots, output_url

router = APIRouter(prefix="/faceless")


@router.get("/voices", response_model=list[VoiceOption])
async def list_voices(
    tts_service: TTSService = Depends(get_tts_service),
) -> list[VoiceOption]:
    return tts_service.list_available_voices()


@router.post("/preview-voice", response_model=VoicePreviewResponse)
async def preview_voice(
    payload: VoicePreviewRequest,
    tts_service: TTSService = Depends(get_tts_service),
) -> VoicePreviewResponse:
    return tts_service.generate_voice_preview(payload.voice, payload.text)


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


@router.post("/generate-animations", response_model=SceneAnimationGenerationResponse)
async def generate_animations(
    payload: SceneAnimationGenerationRequest,
    ai_animation_service: AIAnimationService = Depends(get_ai_animation_service),
) -> SceneAnimationGenerationResponse:
    return ai_animation_service.generate_scene_animations(payload)


@router.post("/generate-ambience", response_model=SceneAmbienceGenerationResponse)
async def generate_ambience(
    payload: SceneAmbienceGenerationRequest,
) -> SceneAmbienceGenerationResponse:
    # Compatibility stage: generated ambience was removed. Rendering uses bundled music assets.
    return SceneAmbienceGenerationResponse(
        job_id=payload.job_id,
        project_id=payload.project_id,
        ambience=[],
    )


@router.post("/generate-music", response_model=StoryMusicGenerationResponse)
async def generate_music(
    payload: StoryMusicGenerationRequest,
    music_service: MusicService = Depends(get_music_service),
) -> StoryMusicGenerationResponse:
    narration = " ".join(scene.narration for scene in payload.scenes)
    mood = payload.mood or music_service.detect_mood(narration)
    music_path = music_service.get_music_for_mood(mood)
    if not music_path:
        from app.core.exceptions import IntegrationError

        raise IntegrationError(f"No bundled music file is available for mood '{mood}'.")
    cache_key = hashlib.sha256(music_path.encode("utf-8")).hexdigest()[:24]
    return StoryMusicGenerationResponse(
        job_id=payload.job_id,
        project_id=payload.project_id,
        prompt=f"Bundled {mood} music asset",
        music_path=music_path,
        music_url=_audio_output_url(music_path),
        duration_seconds=payload.duration_seconds or 0.0,
        cache_key=cache_key,
        cached=True,
    )


@router.post("/render", response_model=StoryRenderResponse)
async def render_story(
    payload: StoryRenderRequest,
    render_service: StoryRenderService = Depends(get_story_render_service),
) -> StoryRenderResponse:
    return render_service.render_story_video(payload)


@router.post("/cleanup-project-output", response_model=ProjectOutputCleanupResponse)
async def cleanup_project_output(
    payload: ProjectOutputCleanupRequest,
) -> ProjectOutputCleanupResponse:
    from app.core.config import get_settings

    settings = get_settings()
    output_roots = iter_project_output_roots(
        output_dir=settings.output_dir,
        project_title=payload.project_title,
        project_id=payload.project_id,
        output_bucket=payload.output_bucket,
    )
    for output_root in output_roots:
        if output_root.exists():
            shutil.rmtree(output_root, ignore_errors=True)

    deleted = all(not output_root.exists() for output_root in output_roots)
    return ProjectOutputCleanupResponse(project_id=payload.project_id, deleted=deleted)


def _audio_output_url(audio_path: str) -> str | None:
    from app.core.config import get_settings

    settings = get_settings()
    resolved_audio_path = Path(audio_path).resolve()
    output_root = Path(settings.output_dir).resolve()
    try:
        resolved_audio_path.relative_to(output_root)
    except ValueError:
        return None
    return output_url(output_dir=settings.output_dir, file_path=resolved_audio_path)
