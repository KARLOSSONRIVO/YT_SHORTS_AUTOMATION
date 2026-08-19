from __future__ import annotations

from dataclasses import dataclass
import logging

from app.core.exceptions import IntegrationError
from app.integrations.pollinations_client import PollinationsClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PollinationsGeneratedImage:
    image_bytes: bytes
    mime_type: str
    provider: str
    model: str


class PollinationsImageGenerationService:
    def __init__(
        self,
        *,
        pollinations_client: PollinationsClient,
        model: str,
        width: int,
        height: int,
    ) -> None:
        self.pollinations_client = pollinations_client
        self.model = model
        self.width = width
        self.height = height

    def is_configured(self) -> bool:
        return self.pollinations_client.is_configured() and bool(self.model)

    def generation_manifest(self) -> dict[str, str]:
        return {
            "provider": "pollinations",
            "model": self.model,
        }

    def generate_image(
        self,
        *,
        prompt: str,
        negative_prompt: str | None = None,
    ) -> PollinationsGeneratedImage:
        if not self.model:
            raise IntegrationError(
                "POLLINATIONS_IMAGE_MODEL is required for Pollinations image generation."
            )

        logger.info("Using Pollinations image generation model %s.", self.model)
        result = self.pollinations_client.generate_image(
            model=self.model,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=self.width,
            height=self.height,
        )
        return PollinationsGeneratedImage(
            image_bytes=result["image_bytes"],
            mime_type=str(result.get("mime_type") or "image/png"),
            provider=str(result.get("provider") or "pollinations"),
            model=self.model,
        )
