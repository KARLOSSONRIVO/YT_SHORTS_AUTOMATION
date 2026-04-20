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
    target_duration_seconds: int = Field(default=45, ge=15, le=180)
    style_preset: str = "cinematic documentary"
    audience: str | None = None


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
    narration: str
    voice: str = "af_sarah"
    speaking_rate: float = Field(default=0.82, gt=0)


class AudioGenerationResponse(BaseModel):
    job_id: str
    project_id: str
    audio_path: str
    audio_url: str
    duration_seconds: float
    voice: str


class VoiceOption(BaseModel):
    voice: str
    label: str
    language: str
    gender: str
    quality_grade: str | None = None
    sample_text: str


class VoicePreviewRequest(BaseModel):
    voice: str
    text: str | None = None


class VoicePreviewResponse(BaseModel):
    voice: str
    audio_path: str
    audio_url: str
    sample_text: str


class SubtitleCue(BaseModel):
    index: int
    start: float
    end: float
    text: str


class StorySubtitleGenerationRequest(BaseModel):
    job_id: str
    project_id: str
    project_title: str | None = None
    audio_path: str | None = None
    scenes: list[FacelessScene] = Field(default_factory=list)


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


class StoryRenderRequest(BaseModel):
    job_id: str
    project_id: str
    project_title: str | None = None
    scenes: list[FacelessScene] = Field(default_factory=list)
    image_paths: list[str] = Field(default_factory=list)
    audio_path: str
    subtitles_path: str | None = None
    background_music_path: str | None = None
    use_music: bool = True
    music_volume: float = Field(default=0.15, ge=0.0, le=1.0)
    ducking: bool = True


class StoryRenderResponse(BaseModel):
    job_id: str
    project_id: str
    video_path: str
    video_url: str
    duration_seconds: float
