from io import BytesIO
from pathlib import Path
import logging

from app.core.exceptions import IntegrationError, PaymentRequiredError
from app.integrations.huggingface_client import HuggingFaceClient
from app.schemas.faceless_video import (
    GeneratedSceneImage,
    SceneImageGenerationRequest,
    SceneImageGenerationResponse,
)
from app.utils.output_paths import output_url, stage_output_dir

logger = logging.getLogger(__name__)


class ImageService:
    COLORS = ["0x22333B", "0x5E503F", "0x2D3142", "0x3A5A40", "0x6D597A", "0x355070"]

    def __init__(
        self,
        *,
        huggingface_client: HuggingFaceClient,
        ffmpeg_client,
        output_dir: str,
        model: str,
        model_path: str | None = None,
        allow_placeholder_generation: bool = False,
    ) -> None:
        self.huggingface_client = huggingface_client
        self.ffmpeg_client = ffmpeg_client
        self.output_dir = Path(output_dir)
        self.model = model
        self.model_path = Path(model_path) if model_path else None
        self.allow_placeholder_generation = allow_placeholder_generation
        self._sdxl_pipeline = None
        # Track whether HF returned 402 so we don't retry it for every scene
        self._hf_payment_required = False

    def generate_scene_images(
        self, payload: SceneImageGenerationRequest
    ) -> SceneImageGenerationResponse:
        stage_dir = stage_output_dir(
            output_dir=self.output_dir,
            output_bucket=payload.output_bucket,
            project_title=payload.project_title,
            project_id=payload.project_id,
            stage_name="scenes",
        )
        stage_dir.mkdir(parents=True, exist_ok=True)
        images: list[GeneratedSceneImage] = []

        for index, scene in enumerate(payload.scenes):
            output_path = stage_dir / f"scene_{scene.scene_index:02d}.png"
            prompt = f"{payload.visual_style}, {scene.image_prompt}, vertical 9:16, no text, no logos"
            try:
                output_path.write_bytes(self._generate_image(prompt))
            except Exception as exc:
                if not self.allow_placeholder_generation:
                    if isinstance(exc, IntegrationError):
                        raise
                    raise IntegrationError(f"Image generation failed: {exc}") from exc
                self._write_placeholder_image(index=index, output_path=output_path)

            images.append(
                GeneratedSceneImage(
                    scene_index=scene.scene_index,
                    prompt=prompt,
                    image_path=str(output_path.resolve()),
                    image_url=output_url(output_dir=self.output_dir, file_path=output_path),
                )
            )

        return SceneImageGenerationResponse(
            job_id=payload.job_id,
            project_id=payload.project_id,
            images=images,
        )

    def _generate_image(self, prompt: str) -> bytes:
        """Try HuggingFace first.  Fall back to local SDXL only on 402."""

        # If HF already returned 402 for this job, skip straight to local
        if not self._hf_payment_required and self.huggingface_client.is_configured():
            try:
                return self.huggingface_client.text_to_image(
                    model=self.model,
                    prompt=prompt,
                )
            except PaymentRequiredError:
                logger.warning(
                    "HuggingFace returned 402 (Payment Required). "
                    "Falling back to local SDXL for remaining images."
                )
                self._hf_payment_required = True
                # Fall through to local SDXL below
            # Any other error (401, 500, network, etc.) → stop immediately
            # IntegrationError will propagate up and stop the pipeline

        # Fallback: local SDXL
        if self._has_local_sdxl():
            return self._generate_local_sdxl(prompt)

        raise IntegrationError(
            "Image generation unavailable: HuggingFace requires payment (402) "
            "and no local SDXL model is configured (PY_WORKER_IMAGE_MODEL_PATH)."
        )

    # ------------------------------------------------------------------ #
    # Local SDXL                                                           #
    # ------------------------------------------------------------------ #

    def _has_local_sdxl(self) -> bool:
        if not self.model_path:
            return False
        return (self.model_path / "model_index.json").exists()

    def _generate_local_sdxl(self, prompt: str) -> bytes:
        try:
            import torch
            from diffusers import StableDiffusionXLPipeline
        except ImportError as exc:
            raise IntegrationError(
                "Local SDXL requires 'diffusers', 'torch', and 'accelerate'. "
                "Add them to requirements.txt and rebuild the Docker image."
            ) from exc

        if self._sdxl_pipeline is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32

            self._sdxl_pipeline = StableDiffusionXLPipeline.from_pretrained(
                str(self.model_path),
                torch_dtype=dtype,
                use_safetensors=True,
                local_files_only=True,
            ).to(device)

            # Memory optimisations when on GPU
            if device == "cuda":
                self._sdxl_pipeline.enable_attention_slicing()

        image = self._sdxl_pipeline(
            prompt=prompt,
            height=1920,
            width=1080,
            num_inference_steps=30,
        ).images[0]

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    # ------------------------------------------------------------------ #
    # Placeholder fallback                                                 #
    # ------------------------------------------------------------------ #

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
