from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    log_level: str = "INFO"
    enable_docs: bool = True
    whisper_model_size: str = "base"
    upload_dir: str = "uploads"
    output_dir: str = "outputs"
    default_min_clip_duration: float = 15.0
    default_max_clip_duration: float = 45.0
    default_top_k: int = 5

    ai_timeout_seconds: float = 120.0
    allow_placeholder_generation: bool = False

    # Gemini is the only generative AI provider. Faster-whisper remains local STT.
    gemini_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("PY_WORKER_GEMINI_API_KEY", "GEMINI_API_KEY"),
    )
    gemini_model: str = Field(
        default="gemini-3.5-flash",
        validation_alias=AliasChoices("PY_WORKER_GEMINI_MODEL", "GEMINI_MODEL"),
    )
    gemini_image_model: str = Field(
        default="gemini-3.1-flash-image",
        validation_alias=AliasChoices("PY_WORKER_GEMINI_IMAGE_MODEL", "GEMINI_IMAGE_MODEL"),
    )
    gemini_tts_model: str = "gemini-3.1-flash-tts-preview"
    gemini_video_model: str = "gemini-omni-flash-preview"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/interactions"

    # Background music
    enable_background_music: bool = True
    default_music_volume: float = 0.15
    enable_audio_ducking: bool = True
    music_assets_path: str = "assets/music"
    reddit_story_background_video_path: str | None = None
    reddit_story_background_music_path: str | None = None


    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PY_WORKER_",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
