from pathlib import Path

from app.core.exceptions import IntegrationError
from app.integrations.huggingface_client import HuggingFaceClient
from app.schemas.faceless_video import (
    GeneratedSceneImage,
    SceneImageGenerationRequest,
    SceneImageGenerationResponse,
)


class ImageService:
    COLORS = ["0x22333B", "0x5E503F", "0x2D3142", "0x3A5A40", "0x6D597A", "0x355070"]

    def __init__(
        self,
        *,
        huggingface_client: HuggingFaceClient,
        ffmpeg_client,
        output_dir: str,
        model: str,
        allow_placeholder_generation: bool = False,
    ) -> None:
        self.huggingface_client = huggingface_client
        self.ffmpeg_client = ffmpeg_client
        self.output_dir = Path(output_dir)
        self.model = model
        self.allow_placeholder_generation = allow_placeholder_generation

    def generate_scene_images(
        self, payload: SceneImageGenerationRequest
    ) -> SceneImageGenerationResponse:
        job_dir = self.output_dir / payload.job_id / "scenes"
        job_dir.mkdir(parents=True, exist_ok=True)
        images: list[GeneratedSceneImage] = []

        for index, scene in enumerate(payload.scenes):
            output_path = job_dir / f"scene_{scene.scene_index:02d}.png"
            prompt = f"{payload.visual_style}, {scene.image_prompt}, vertical 9:16, no text, no logos"
            try:
                output_path.write_bytes(
                    self.huggingface_client.text_to_image(
                        model=self.model,
                        prompt=prompt,
                    )
                )
            except Exception as exc:
                if not self.allow_placeholder_generation:
                    if isinstance(exc, IntegrationError):
                        raise
                    raise IntegrationError(f"Hugging Face image generation failed: {exc}") from exc
                self._write_placeholder_image(index=index, output_path=output_path)

            images.append(
                GeneratedSceneImage(
                    scene_index=scene.scene_index,
                    prompt=prompt,
                    image_path=str(output_path.resolve()),
                    image_url=f"/outputs/{payload.job_id}/scenes/{output_path.name}",
                )
            )

        return SceneImageGenerationResponse(
            job_id=payload.job_id,
            project_id=payload.project_id,
            images=images,
        )

    def _write_placeholder_image(self, *, index: int, output_path: Path) -> None:
        if not self.ffmpeg_client.is_available():
            raise IntegrationError("ffmpeg is required to create placeholder scene images.")

        color = self.COLORS[index % len(self.COLORS)]
        self.ffmpeg_client.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"color=c={color}:s=1080x1920",
                "-frames:v",
                "1",
                str(output_path),
            ]
        )
