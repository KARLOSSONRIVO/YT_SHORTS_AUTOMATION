from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.integrations.gemini_client import GeminiClient
from app.schemas.faceless_video import (
    GeneratedSceneAnimation,
    SceneAnimationGenerationRequest,
    SceneAnimationGenerationResponse,
)
from app.utils.output_paths import output_url, stage_output_dir


class AIAnimationService:
    def __init__(self, *, gemini_client: GeminiClient, output_dir: str, model: str) -> None:
        self.gemini_client = gemini_client
        self.output_dir = Path(output_dir)
        self.model = model
        self.cache_dir = self.output_dir / ".gemini-video-cache"

    def generate_scene_animations(self, payload: SceneAnimationGenerationRequest) -> SceneAnimationGenerationResponse:
        stage_dir = stage_output_dir(
            output_dir=self.output_dir,
            output_bucket=payload.output_bucket,
            project_title=payload.project_title,
            project_id=payload.project_id,
            stage_name="animations",
        )
        stage_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        image_by_scene = {image.scene_index: image.image_path for image in payload.images}
        animations: list[GeneratedSceneAnimation] = []

        for scene in payload.scenes:
            prompt = self.build_animation_prompt(scene_data=scene, animation_style=payload.animation_style)
            cache_key = self._cache_key(prompt)
            cached_path = self.cache_dir / f"{cache_key}.mp4"
            output_path = stage_dir / f"scene_{scene.scene_index:02d}_animation.mp4"
            if not cached_path.exists():
                cached_path.write_bytes(
                    self.gemini_client.generate_video(model=self.model, prompt=prompt, aspect_ratio="9:16")
                )
                cached_path.with_suffix(".json").write_text(
                    json.dumps({"model": self.model, "prompt": prompt}, indent=2), encoding="utf-8"
                )
            output_path.write_bytes(cached_path.read_bytes())
            animations.append(
                GeneratedSceneAnimation(
                    scene_index=scene.scene_index,
                    prompt=prompt,
                    source_image_path=image_by_scene.get(scene.scene_index, ""),
                    video_path=str(output_path.resolve()),
                    video_url=output_url(output_dir=self.output_dir, file_path=output_path),
                    cache_key=cache_key,
                )
            )
        return SceneAnimationGenerationResponse(
            job_id=payload.job_id, project_id=payload.project_id, animations=animations
        )

    def build_animation_prompt(self, *, scene_data: Any, animation_style: str | None) -> str:
        parts = [
            animation_style or "cinematic story animation",
            getattr(scene_data, "image_prompt", ""),
            getattr(scene_data, "narration", ""),
            "natural subject motion, atmospheric camera movement, vertical 9:16 composition",
            "no text, no subtitles, no logos, no watermarks",
        ]
        return ", ".join(" ".join(part.split()) for part in parts if part)

    def _cache_key(self, prompt: str) -> str:
        return hashlib.sha256(f"{self.model}\n{prompt}".encode("utf-8")).hexdigest()[:24]
