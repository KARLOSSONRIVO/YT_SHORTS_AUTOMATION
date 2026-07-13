from functools import lru_cache

from app.core.config import Settings, get_settings
from app.integrations.ffprobe_client import FFprobeClient
from app.integrations.ffmpeg_client import FFmpegClient
from app.integrations.gemini_client import GeminiClient
from app.integrations.whisper_client import WhisperClient
from app.pipelines.highlight_pipeline import HighlightPipeline
from app.pipelines.subtitle_pipeline import SubtitlePipeline
from app.pipelines.transcription_pipeline import TranscriptionPipeline
from app.pipelines.workflow_pipeline import WorkflowPipeline
from app.services.clip_detection_service import ClipDetectionService
from app.services.clip_ranking_service import ClipRankingService
from app.services.hook_scoring_service import HookScoringService
from app.services.faceless_subtitle_service import FacelessSubtitleService
from app.services.ai_animation_service import AIAnimationService
from app.services.image_service import ImageService
from app.services.gemini_image_generation_service import GeminiImageGenerationService
from app.services.local_media_store_service import LocalMediaStoreService
from app.services.music_service import MusicService
from app.services.keyword_scoring_service import KeywordScoringService
from app.services.llm_service import LLMService
from app.services.media_prep_service import MediaPrepService
from app.services.metadata_service import MetadataService
from app.services.render_service import RenderService
from app.services.silence_detection_service import SilenceDetectionService
from app.services.speech_intensity_service import SpeechIntensityService
from app.services.story_render_service import StoryRenderService
from app.services.subtitle_formatting_service import SubtitleFormattingService
from app.services.tts_service import TTSService
from app.services.transcript_segmentation_service import TranscriptSegmentationService
from app.services.transcription_service import TranscriptionService


@lru_cache
def get_ffprobe_client() -> FFprobeClient:
    return FFprobeClient()


@lru_cache
def get_whisper_client() -> WhisperClient:
    settings = get_settings()
    return WhisperClient(model_name=settings.whisper_model_size)


@lru_cache
def get_ffmpeg_client() -> FFmpegClient:
    return FFmpegClient()


@lru_cache
def get_gemini_client() -> GeminiClient:
    settings = get_settings()
    return GeminiClient(
        api_key=settings.gemini_api_key,
        base_url=settings.gemini_base_url,
        timeout_seconds=settings.ai_timeout_seconds,
    )


@lru_cache
def get_gemini_image_generation_service() -> GeminiImageGenerationService:
    settings = get_settings()
    return GeminiImageGenerationService(
        gemini_client=get_gemini_client(),
        model=settings.gemini_image_model,
    )


@lru_cache
def get_metadata_service() -> MetadataService:
    return MetadataService(ffprobe_client=get_ffprobe_client())


@lru_cache
def get_media_prep_service() -> MediaPrepService:
    return MediaPrepService()


@lru_cache
def get_transcription_service() -> TranscriptionService:
    return TranscriptionService(whisper_client=get_whisper_client())


@lru_cache
def get_subtitle_formatting_service() -> SubtitleFormattingService:
    return SubtitleFormattingService()


@lru_cache
def get_transcript_segmentation_service() -> TranscriptSegmentationService:
    return TranscriptSegmentationService()


@lru_cache
def get_silence_detection_service() -> SilenceDetectionService:
    return SilenceDetectionService()


@lru_cache
def get_hook_scoring_service() -> HookScoringService:
    return HookScoringService()


@lru_cache
def get_keyword_scoring_service() -> KeywordScoringService:
    return KeywordScoringService()


@lru_cache
def get_speech_intensity_service() -> SpeechIntensityService:
    return SpeechIntensityService()


@lru_cache
def get_clip_detection_service() -> ClipDetectionService:
    return ClipDetectionService()


@lru_cache
def get_clip_ranking_service() -> ClipRankingService:
    return ClipRankingService()


def get_settings_dep() -> Settings:
    return get_settings()


@lru_cache
def get_local_media_store_service() -> LocalMediaStoreService:
    settings = get_settings()
    return LocalMediaStoreService(upload_dir=settings.upload_dir)


@lru_cache
def get_render_service() -> RenderService:
    settings = get_settings()
    return RenderService(
        ffmpeg_client=get_ffmpeg_client(),
        output_dir=settings.output_dir,
    )


@lru_cache
def get_llm_service() -> LLMService:
    settings = get_settings()
    return LLMService(
        llm_client=get_gemini_client(),
        model=settings.gemini_model,
        allow_placeholder_generation=settings.allow_placeholder_generation,
    )


@lru_cache
def get_tts_service() -> TTSService:
    settings = get_settings()
    return TTSService(
        gemini_client=get_gemini_client(),
        ffmpeg_client=get_ffmpeg_client(),
        output_dir=settings.output_dir,
        model=settings.gemini_tts_model,
        allow_placeholder_generation=settings.allow_placeholder_generation,
    )


@lru_cache
def get_faceless_subtitle_service() -> FacelessSubtitleService:
    settings = get_settings()
    return FacelessSubtitleService(
        output_dir=settings.output_dir,
        whisper_client=get_whisper_client(),
        allow_placeholder_generation=settings.allow_placeholder_generation,
    )


@lru_cache
def get_image_service() -> ImageService:
    settings = get_settings()
    return ImageService(
        ffmpeg_client=get_ffmpeg_client(),
        output_dir=settings.output_dir,
        gemini_image_generation_service=get_gemini_image_generation_service(),
        allow_placeholder_generation=settings.allow_placeholder_generation,
    )


@lru_cache
def get_story_render_service() -> StoryRenderService:
    settings = get_settings()
    return StoryRenderService(
        ffmpeg_client=get_ffmpeg_client(),
        output_dir=settings.output_dir,
        music_service=get_music_service(),
        llm_service=get_llm_service(),
        enable_background_music=settings.enable_background_music,
        default_music_volume=settings.default_music_volume,
        enable_audio_ducking=settings.enable_audio_ducking,
        reddit_story_background_video_path=settings.reddit_story_background_video_path,
        reddit_story_background_music_path=settings.reddit_story_background_music_path,
    )


@lru_cache
def get_music_service() -> MusicService:
    settings = get_settings()
    return MusicService(music_assets_path=settings.music_assets_path)


@lru_cache
def get_ai_animation_service() -> AIAnimationService:
    settings = get_settings()
    return AIAnimationService(
        gemini_client=get_gemini_client(),
        output_dir=settings.output_dir,
        model=settings.gemini_video_model,
    )


def get_transcription_pipeline() -> TranscriptionPipeline:
    return TranscriptionPipeline(
        metadata_service=get_metadata_service(),
        media_prep_service=get_media_prep_service(),
        transcription_service=get_transcription_service(),
    )


def get_subtitle_pipeline() -> SubtitlePipeline:
    return SubtitlePipeline(
        subtitle_formatting_service=get_subtitle_formatting_service(),
    )


def get_highlight_pipeline() -> HighlightPipeline:
    return HighlightPipeline(
        metadata_service=get_metadata_service(),
        media_prep_service=get_media_prep_service(),
        transcription_service=get_transcription_service(),
        transcript_segmentation_service=get_transcript_segmentation_service(),
        silence_detection_service=get_silence_detection_service(),
        hook_scoring_service=get_hook_scoring_service(),
        keyword_scoring_service=get_keyword_scoring_service(),
        speech_intensity_service=get_speech_intensity_service(),
        clip_detection_service=get_clip_detection_service(),
        clip_ranking_service=get_clip_ranking_service(),
        subtitle_formatting_service=get_subtitle_formatting_service(),
    )


def get_workflow_pipeline() -> WorkflowPipeline:
    return WorkflowPipeline(
        highlight_pipeline=get_highlight_pipeline(),
        render_service=get_render_service(),
    )
