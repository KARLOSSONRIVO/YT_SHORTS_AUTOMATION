from pydantic import BaseModel, Field


class FacelessScene(BaseModel):
    scene_index: int
    narration: str
    image_prompt: str
    duration_seconds: float = Field(default=6.0, gt=0)
    caption_text: str


class ScriptGenerationRequest(BaseModel):
    job_id: str
    project_id: str
    project_title: str | None = None
    topic: str
    tone: str = "dramatic"
    language: str = "en"
    target_duration_seconds: int = Field(default=60, ge=15, le=180)
    style_preset: str = "cinematic documentary"
    audience: str | None = None
    script_framework: str = "psychology_truth"
    story_format: str | None = None
    speaking_rate: float = Field(default=1.0, ge=0.5, le=2.0)


class ScriptGenerationResponse(BaseModel):
    job_id: str
    project_id: str
    title: str
    hook: str
    narration: str
    scenes: list[FacelessScene] = Field(default_factory=list)
    image_prompts: list[str] = Field(default_factory=list)
    caption_text: str


class AudioGenerationRequest(BaseModel):
    job_id: str
    project_id: str
    project_title: str | None = None
    output_bucket: str | None = None
    narration: str
    voice: str = "Kore"
    speaking_rate: float = Field(default=0.82, gt=0)


class AudioGenerationResponse(BaseModel):
    job_id: str
    project_id: str
    audio_path: str
    audio_url: str
    duration_seconds: float
    voice: str


class SubtitleCue(BaseModel):
    index: int
    start: float
    end: float
    text: str


class StorySubtitleGenerationRequest(BaseModel):
    job_id: str
    project_id: str
    project_title: str | None = None
    opening_display_text: str | None = None
    output_bucket: str | None = None
    audio_path: str | None = None
    scenes: list[FacelessScene] = Field(default_factory=list)
    font_family: str | None = None
    font_size: int | None = Field(default=None, ge=24, le=120)
    fill_color: str | None = None
    stroke_color: str | None = None
    highlight_color: str | None = None
    position: str | None = None
    max_chars_per_line: int | None = Field(default=None, ge=12, le=42)
    max_lines: int | None = Field(default=None, ge=1, le=4)


class StorySubtitleGenerationResponse(BaseModel):
    job_id: str
    project_id: str
    srt_path: str
    ass_path: str
    timestamp_json_path: str
    srt_url: str
    ass_url: str
    timestamp_json_url: str
    subtitles: list[SubtitleCue] = Field(default_factory=list)


class SceneImageGenerationRequest(BaseModel):
    job_id: str
    project_id: str
    project_title: str | None = None
    output_bucket: str | None = None
    scenes: list[FacelessScene] = Field(default_factory=list)
    visual_style: str = "vertical cinematic, high contrast"


class GeneratedSceneImage(BaseModel):
    scene_index: int
    prompt: str
    image_path: str
    image_url: str


class SceneImageGenerationResponse(BaseModel):
    job_id: str
    project_id: str
    images: list[GeneratedSceneImage] = Field(default_factory=list)


class SceneAnimationGenerationRequest(BaseModel):
    job_id: str
    project_id: str
    project_title: str | None = None
    output_bucket: str | None = None
    scenes: list[FacelessScene] = Field(default_factory=list)
    images: list[GeneratedSceneImage] = Field(default_factory=list)
    animation_style: str = "cinematic story animation"
    num_frames: int | None = Field(default=None, ge=8, le=160)
    num_inference_steps: int | None = Field(default=None, ge=1, le=80)
    guidance_scale: float | None = Field(default=None, ge=0.0, le=20.0)


class GeneratedSceneAnimation(BaseModel):
    scene_index: int
    prompt: str
    source_image_path: str
    video_path: str
    video_url: str
    cache_key: str


class SceneAnimationGenerationResponse(BaseModel):
    job_id: str
    project_id: str
    animations: list[GeneratedSceneAnimation] = Field(default_factory=list)


class StoryRenderRequest(BaseModel):
    job_id: str
    project_id: str
    project_title: str | None = None
    output_bucket: str | None = None
    scenes: list[FacelessScene] = Field(default_factory=list)
    image_paths: list[str] = Field(default_factory=list)
    scene_video_paths: list[str] = Field(default_factory=list)
    audio_path: str
    subtitles_path: str | None = None
    background_music_path: str | None = None
    render_mode: str = "scene_images"
    animation_style: str | None = None
    animation_intensity: float = Field(default=1.0, ge=0.25, le=2.0)
    use_music: bool = True
    music_volume: float = Field(default=0.15, ge=0.0, le=1.0)
    narration_volume: float = Field(default=1.0, ge=0.0, le=2.0)
    ambience_audio_paths: list[str] = Field(default_factory=list)
    ambience_volume: float = Field(default=0.08, ge=0.0, le=1.0)
    sfx_audio_paths: list[str] = Field(default_factory=list)
    ducking: bool = True


class StoryRenderResponse(BaseModel):
    job_id: str
    project_id: str
    video_path: str
    video_url: str
    duration_seconds: float


class ProjectOutputCleanupRequest(BaseModel):
    project_id: str
    project_title: str | None = None
    output_bucket: str | None = None


class ProjectOutputCleanupResponse(BaseModel):
    project_id: str
    deleted: bool


class SceneAmbienceGenerationRequest(BaseModel):
    job_id: str
    project_id: str
    project_title: str | None = None
    output_bucket: str | None = None
    scenes: list[FacelessScene] = Field(default_factory=list)
    output_format: str = "wav"
    duration_seconds: float | None = Field(default=None, ge=1.0, le=60.0)


class GeneratedSceneAmbience(BaseModel):
    scene_index: int
    prompt: str
    audio_path: str
    audio_url: str | None = None
    duration_seconds: float
    cache_key: str
    cached: bool
    mood: str
    environment: str
    emotional_tone: str
    tension_level: float


class SceneAmbienceGenerationResponse(BaseModel):
    job_id: str
    project_id: str
    ambience: list[GeneratedSceneAmbience] = Field(default_factory=list)


class StoryMusicGenerationRequest(BaseModel):
    job_id: str
    project_id: str
    project_title: str | None = None
    output_bucket: str | None = None
    scenes: list[FacelessScene] = Field(default_factory=list)
    prompt: str | None = None
    mood: str | None = None
    output_format: str = "wav"
    duration_seconds: float | None = Field(default=None, ge=5.0, le=120.0)


class StoryMusicGenerationResponse(BaseModel):
    job_id: str
    project_id: str
    prompt: str
    music_path: str
    music_url: str | None = None
    duration_seconds: float
    cache_key: str
    cached: bool
