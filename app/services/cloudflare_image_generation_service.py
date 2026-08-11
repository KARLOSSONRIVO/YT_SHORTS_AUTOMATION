from __future__ import annotations

from dataclasses import dataclass
import logging

from app.core.exceptions import IntegrationError
from app.integrations.cloudflare_workers_ai_client import CloudflareWorkersAIClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CloudflareGeneratedImage:
    image_bytes: bytes
    mime_type: str
    provider: str
    model: str


class CloudflareImageGenerationService:
    def __init__(
        self,
        *,
        cloudflare_client: CloudflareWorkersAIClient,
        model: str,
        width: int,
        height: int,
        num_steps: int,
        guidance: float,
    ) -> None:
        self.cloudflare_client = cloudflare_client
        self.model = model
        self.width = width
        self.height = height
        self.num_steps = num_steps
        self.guidance = guidance

    def is_configured(self) -> bool:
        return self.cloudflare_client.is_configured() and bool(self.model)

    def generate_image(
        self,
        *,
        prompt: str,
        negative_prompt: str | None = None,
    ) -> CloudflareGeneratedImage:
        if not self.model:
            raise IntegrationError(
                "CLOUDFLARE_IMAGE_MODEL is required for Cloudflare Workers AI "
                "image generation."
            )

        logger.info("Using Cloudflare Workers AI image generation model %s.", self.model)
        result = self.cloudflare_client.generate_image(
            model=self.model,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=self.width,
            height=self.height,
            num_steps=self.num_steps,
            guidance=self.guidance,
        )
        return CloudflareGeneratedImage(
            image_bytes=result["image_bytes"],
            mime_type=str(result.get("mime_type") or "image/png"),
            provider=str(result.get("provider") or "cloudflare_workers_ai"),
            model=self.model,
        )
