from __future__ import annotations

from dataclasses import dataclass
import logging

from app.core.exceptions import IntegrationError
from app.integrations.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeminiGeneratedImage:
    image_bytes: bytes
    mime_type: str
    provider: str
    model: str


class GeminiImageGenerationService:
    def __init__(
        self,
        *,
        gemini_client: GeminiClient,
        model: str,
    ) -> None:
        self.gemini_client = gemini_client
        self.model = model

    def is_configured(self) -> bool:
        return self.gemini_client.is_configured() and bool(self.model)

    def generate_image(self, *, prompt: str) -> GeminiGeneratedImage:
        if not self.model:
            raise IntegrationError("GEMINI_IMAGE_MODEL is required for Gemini image generation.")
        if not self.gemini_client.is_configured():
            raise IntegrationError("GEMINI_API_KEY is required for Gemini image generation.")

        logger.info("Using Gemini image generation.")
        result = self.gemini_client.generate_image(model=self.model, prompt=prompt)
        return GeminiGeneratedImage(
            image_bytes=result["image_bytes"],
            mime_type=str(result.get("mime_type") or "image/png"),
            provider=str(result.get("provider") or "gemini"),
            model=self.model,
        )
