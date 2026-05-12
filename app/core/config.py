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
    hf_timeout_seconds: float = 0.0
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
    image_model: str = "black-forest-labs/FLUX.1-dev"
    image_model_path: str | None = None  # Local fallback path remains SDXL-specific unless the image service is expanded

    # TTS (Kokoro)
    tts_model_path: str | None = None
    tts_model: str = "hexgrad/Kokoro-82M"

    # Background music
    enable_background_music: bool = True
    default_music_volume: float = 0.15
    enable_audio_ducking: bool = True
    music_assets_path: str = "assets/music"
    reddit_story_background_video_path: str | None = None
    reddit_story_background_music_path: str | None = None

    # AI-generated cinematic ambience (optional)
    enable_ai_ambience: bool = False
    ai_audio_model: str = "stable-audio-open"
    ai_audio_cache_dir: str = "assets/generated_audio"
    ai_ambience_volume: float = 0.08
    ai_audio_duration_seconds: float = 12.0
    ai_audio_inference_steps: int = 100

    # AI-generated background music (optional, Hugging Face API)
    enable_ai_music: bool = False
    ai_music_model: str = "musicgen-small"
    ai_music_cache_dir: str = "assets/generated_music"
    ai_music_duration_seconds: float = 30.0

    # AI-generated scene animation (optional, Hugging Face API)
    enable_ai_animation: bool = False
    ai_animation_model: str = "Wan-AI/Wan2.2-I2V-A14B"
    ai_animation_cache_dir: str = "assets/generated_animations"
    ai_animation_num_frames: int = 81
    ai_animation_inference_steps: int = 30
    ai_animation_guidance_scale: float = 5.0

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
