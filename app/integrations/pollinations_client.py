from __future__ import annotations

import base64
import binascii
import json
import logging
from typing import Any

import httpx

from app.core.exceptions import (
    IntegrationError,
    PaymentRequiredError,
    ProviderRateLimitError,
)

logger = logging.getLogger(__name__)


class PollinationsClient:
    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str,
        timeout_seconds: float | None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.api_key = api_key.strip() if api_key else None
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = (
            timeout_seconds if timeout_seconds and timeout_seconds > 0 else None
        )
        self.transport = transport

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def generate_image(
        self,
        *,
        model: str,
        prompt: str,
        negative_prompt: str | None = None,
        width: int = 768,
        height: int = 1024,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise IntegrationError(
                "POLLINATIONS_API_KEY is required for Pollinations image generation."
            )
        if not model.strip():
            raise IntegrationError(
                "POLLINATIONS_IMAGE_MODEL is required for Pollinations image generation."
            )
        if not prompt.strip():
            raise IntegrationError("An image prompt is required for Pollinations.")

        generation_prompt = prompt.strip()
        if negative_prompt and negative_prompt.strip():
            generation_prompt += f"\nAvoid: {negative_prompt.strip()}"

        payload = {
            "prompt": generation_prompt,
            "model": model,
            "n": 1,
            "size": f"{width}x{height}",
            "response_format": "b64_json",
        }
        endpoint = f"{self.base_url}/images/generations"

        try:
            with httpx.Client(
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = client.post(
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
        except httpx.RequestError as exc:
            raise IntegrationError(
                f"Pollinations image generation request failed: {exc}"
            ) from exc

        if response.status_code >= 400:
            message = self._error_message(response)
            logger.error(
                "Pollinations image generation failed. model=%s status=%s message=%s",
                model,
                response.status_code,
                message,
            )
            if response.status_code == 429:
                raise ProviderRateLimitError(message)
            if response.status_code == 402:
                raise PaymentRequiredError(message)
            raise IntegrationError(message)

        try:
            body = response.json()
        except json.JSONDecodeError as exc:
            raise IntegrationError(
                "Pollinations image generation returned a non-JSON response."
            ) from exc

        encoded = self._encoded_image(body)
        if not encoded:
            raise IntegrationError(
                "Pollinations image generation returned no image data."
            )

        try:
            image_bytes = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise IntegrationError(
                "Pollinations image generation returned no valid image data."
            ) from exc

        if not image_bytes:
            raise IntegrationError(
                "Pollinations image generation returned no valid image data."
            )

        return {
            "image_bytes": image_bytes,
            "mime_type": "image/png",
            "provider": "pollinations",
        }

    def generate_speech(
        self,
        *,
        model: str,
        text: str,
        voice: str,
        instruct: str | None,
        speed: float,
        seed: int,
    ) -> bytes:
        if not self.api_key:
            raise IntegrationError(
                "POLLINATIONS_API_KEY is required for Pollinations speech generation."
            )
        if not model.strip():
            raise IntegrationError(
                "POLLINATIONS_TTS_MODEL is required for Pollinations speech generation."
            )
        if not text.strip():
            raise IntegrationError("Narration text is required for Pollinations speech.")

        payload: dict[str, Any] = {
            "model": model.strip(),
            "input": text.strip(),
            "voice": voice.strip() or "alloy",
            "response_format": "wav",
            "speed": speed,
            "seed": seed,
        }
        if instruct and instruct.strip():
            payload["instruct"] = instruct.strip()
        endpoint = f"{self.base_url}/audio/speech"

        try:
            with httpx.Client(
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = client.post(
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
        except httpx.RequestError as exc:
            raise IntegrationError(
                f"Pollinations speech generation request failed: {exc}"
            ) from exc

        if response.status_code >= 400:
            message = self._error_message(response, operation="speech generation")
            logger.error(
                "Pollinations speech generation failed. model=%s status=%s message=%s",
                model,
                response.status_code,
                message,
            )
            if response.status_code == 429:
                raise ProviderRateLimitError(message)
            if response.status_code == 402:
                raise PaymentRequiredError(message)
            raise IntegrationError(message)

        if not response.content:
            raise IntegrationError(
                "Pollinations speech generation returned no audio data."
            )
        return response.content

    @staticmethod
    def _encoded_image(body: Any) -> str | None:
        if not isinstance(body, dict):
            return None
        data = body.get("data")
        if not isinstance(data, list) or not data or not isinstance(data[0], dict):
            return None
        encoded = data[0].get("b64_json")
        return encoded if isinstance(encoded, str) else None

    @staticmethod
    def _error_message(
        response: httpx.Response,
        *,
        operation: str = "image generation",
    ) -> str:
        try:
            body = response.json()
        except json.JSONDecodeError:
            body = response.text

        detail: Any = body
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                detail = error.get("message") or error
            elif error:
                detail = error
        return (
            f"Pollinations {operation} failed with status "
            f"{response.status_code}: {detail}"
        )
