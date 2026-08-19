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

    # Groq generates story scripts, Pollinations generates scene images, and
    # Gemini generates speech and video.
    groq_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("PY_WORKER_GROQ_API_KEY", "GROQ_API_KEY"),
    )
    groq_model: str = Field(
        default="qwen/qwen3.6-27b",
        validation_alias=AliasChoices("PY_WORKER_GROQ_MODEL", "GROQ_MODEL"),
    )
    groq_fallback_model: str | None = Field(
        default="openai/gpt-oss-20b",
        validation_alias=AliasChoices(
            "PY_WORKER_GROQ_FALLBACK_MODEL",
            "GROQ_FALLBACK_MODEL",
        ),
    )
    groq_base_url: str = Field(
        default="https://api.groq.com/openai/v1",
        validation_alias=AliasChoices("PY_WORKER_GROQ_BASE_URL", "GROQ_BASE_URL"),
    )

    gemini_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("PY_WORKER_GEMINI_API_KEY", "GEMINI_API_KEY"),
    )
    gemini_model: str = Field(
        default="gemini-3.5-flash",
        validation_alias=AliasChoices("PY_WORKER_GEMINI_MODEL", "GEMINI_MODEL"),
    )
    pollinations_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "PY_WORKER_POLLINATIONS_API_KEY",
            "POLLINATIONS_API_KEY",
        ),
    )
    pollinations_image_model: str = Field(
        default="flux",
        validation_alias=AliasChoices(
            "PY_WORKER_POLLINATIONS_IMAGE_MODEL",
            "POLLINATIONS_IMAGE_MODEL",
        ),
    )
    pollinations_image_width: int = Field(
        default=768,
        validation_alias=AliasChoices(
            "PY_WORKER_POLLINATIONS_IMAGE_WIDTH",
            "POLLINATIONS_IMAGE_WIDTH",
        ),
    )
    pollinations_image_height: int = Field(
        default=1024,
        validation_alias=AliasChoices(
            "PY_WORKER_POLLINATIONS_IMAGE_HEIGHT",
            "POLLINATIONS_IMAGE_HEIGHT",
        ),
    )
    pollinations_base_url: str = Field(
        default="https://gen.pollinations.ai/v1",
        validation_alias=AliasChoices(
            "PY_WORKER_POLLINATIONS_BASE_URL",
            "POLLINATIONS_BASE_URL",
        ),
    )
    pollinations_image_timeout_seconds: float = 300.0
    pollinations_tts_model: str = Field(
        default="elevenlabs",
        validation_alias=AliasChoices(
            "PY_WORKER_POLLINATIONS_TTS_MODEL",
            "POLLINATIONS_TTS_MODEL",
        ),
    )
    gemini_tts_model: str = "gemini-3.1-flash-tts-preview"
    gemini_video_model: str = "gemini-omni-flash-preview"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/interactions"

    # Background music
    enable_background_music: bool = True
    default_music_volume: float = 0.15
    enable_audio_ducking: bool = True
    music_assets_path: str = "assets/music"
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PY_WORKER_",
        env_ignore_empty=True,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
