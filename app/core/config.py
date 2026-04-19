from functools import lru_cache

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

    # HuggingFace (remote inference fallback)
    hf_token: str | None = None
    hf_inference_base_url: str = "https://api-inference.huggingface.co/models"
    hf_router_base_url: str = "https://router.huggingface.co/v1"
    hf_timeout_seconds: float = 600.0
    allow_placeholder_generation: bool = False

    # LLM — HuggingFace (remote) settings
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str = "Qwen/Qwen3-4B-Instruct-2507"

    # LLM — Ollama (local) settings
    use_ollama_for_llm: bool = False
    ollama_base_url: str = "http://host.docker.internal:11434/v1"
    ollama_llm_model: str = "qwen2.5:4b"

    # Image generation
    image_model_base_url: str | None = None
    image_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    image_model_path: str | None = None  # Set to local SDXL path to run offline

    # TTS (Kokoro)
    tts_model_path: str | None = None
    tts_model: str = "hexgrad/Kokoro-82M"

    # Background music
    enable_background_music: bool = True
    default_music_volume: float = 0.15
    enable_audio_ducking: bool = True
    music_assets_path: str = "assets/music"

    # Whisper (STT)
    whisper_hf_model: str = "openai/whisper-large-v3-turbo"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PY_WORKER_",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
