from __future__ import annotations

import base64
import binascii
import json
import logging
from typing import Any

import httpx

from app.core.exceptions import (
    ContentSafetyError,
    IntegrationError,
    ProviderRateLimitError,
)

logger = logging.getLogger(__name__)


class CloudflareWorkersAIClient:
    def __init__(
        self,
        *,
        account_id: str | None,
        api_token: str | None,
        base_url: str,
        timeout_seconds: float | None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.account_id = account_id.strip() if account_id else None
        self.api_token = api_token.strip() if api_token else None
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = (
            timeout_seconds if timeout_seconds and timeout_seconds > 0 else None
        )
        self.transport = transport

    def is_configured(self) -> bool:
        return bool(self.account_id and self.api_token)

    def generate_image(
        self,
        *,
        model: str,
        prompt: str,
        negative_prompt: str | None = None,
        width: int = 1024,
        height: int = 1792,
        num_steps: int = 8,
        guidance: float = 7.5,
    ) -> dict[str, Any]:
        if not self.account_id or not self.api_token:
            raise IntegrationError(
                "CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN are required "
                "for Cloudflare Workers AI image generation."
            )
        if not model:
            raise IntegrationError(
                "CLOUDFLARE_IMAGE_MODEL is required for Cloudflare Workers AI "
                "image generation."
            )
        if not prompt.strip():
            raise IntegrationError("An image prompt is required for Cloudflare Workers AI.")

        payload: dict[str, Any] = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_steps": num_steps,
            "guidance": guidance,
        }
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

        auth_headers = {"Authorization": f"Bearer {self.api_token}"}
        endpoint = f"{self.base_url}/{self.account_id}/ai/run/{model}"

        try:
            with httpx.Client(
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                if model.startswith("@cf/black-forest-labs/flux-2-"):
                    flux_prompt = prompt.strip()
                    if negative_prompt and negative_prompt.strip():
                        flux_prompt += f"\nAvoid: {negative_prompt.strip()}"
                    files = {
                        "prompt": (None, flux_prompt),
                        "width": (None, str(width)),
                        "height": (None, str(height)),
                        "guidance": (None, str(guidance)),
                    }
                    if model == "@cf/black-forest-labs/flux-2-dev":
                        files["steps"] = (None, str(num_steps))
                    response = client.post(
                        endpoint,
                        headers=auth_headers,
                        files=files,
                    )
                else:
                    response = client.post(
                        endpoint,
                        headers={
                            **auth_headers,
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
        except httpx.RequestError as exc:
            raise IntegrationError(
                f"Cloudflare Workers AI image request failed: {exc}"
            ) from exc

        if response.status_code >= 400:
            message = self._error_message(response)
            logger.error(
                "Cloudflare Workers AI image generation failed. "
                "model=%s status=%s message=%s",
                model,
                response.status_code,
                message,
            )
            if response.status_code == 429:
                raise ProviderRateLimitError(message)
            if self._is_content_safety_rejection(response=response, message=message):
                raise ContentSafetyError(message)
            raise IntegrationError(message)

        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip()
        if content_type.startswith("image/"):
            if not response.content:
                raise IntegrationError(
                    "Cloudflare Workers AI image generation returned an empty image."
                )
            return {
                "image_bytes": response.content,
                "mime_type": content_type,
                "provider": "cloudflare_workers_ai",
            }

        try:
            body = response.json()
        except json.JSONDecodeError:
            if response.content:
                return {
                    "image_bytes": response.content,
                    "mime_type": content_type or "image/png",
                    "provider": "cloudflare_workers_ai",
                }
            raise IntegrationError(
                "Cloudflare Workers AI image generation returned an empty response."
            )

        if isinstance(body, dict) and body.get("success") is False:
            raise IntegrationError(self._body_error_message(body, response.status_code))

        image_bytes, mime_type = self._extract_image(body)
        if not image_bytes:
            raise IntegrationError(
                "Cloudflare Workers AI image generation returned no image output."
            )
        return {
            "image_bytes": image_bytes,
            "mime_type": mime_type,
            "provider": "cloudflare_workers_ai",
        }

    def _extract_image(self, body: Any) -> tuple[bytes | None, str]:
        value = body.get("result") if isinstance(body, dict) else body
        mime_type = "image/png"

        if isinstance(value, dict):
            mime_type = str(
                value.get("mime_type")
                or value.get("mimeType")
                or value.get("content_type")
                or "image/png"
            )
            encoded = (
                value.get("image")
                or value.get("image_b64")
                or value.get("base64")
                or value.get("data")
            )
        else:
            encoded = value

        if not isinstance(encoded, str) or not encoded.strip():
            return None, mime_type

        if encoded.startswith("data:") and ";base64," in encoded:
            header, encoded = encoded.split(",", 1)
            declared_type = header[5:].split(";", 1)[0]
            if declared_type:
                mime_type = declared_type

        try:
            return base64.b64decode(encoded, validate=True), mime_type
        except (binascii.Error, ValueError) as exc:
            raise IntegrationError(
                "Cloudflare Workers AI returned invalid base64 image data."
            ) from exc

    def _error_message(self, response: httpx.Response) -> str:
        try:
            body = response.json()
        except json.JSONDecodeError:
            body = response.text
        if isinstance(body, dict):
            return self._body_error_message(body, response.status_code)
        return (
            "Cloudflare Workers AI image generation failed with status "
            f"{response.status_code}: {body}"
        )

    def _body_error_message(self, body: dict[str, Any], status_code: int) -> str:
        errors = body.get("errors")
        if isinstance(errors, list) and errors:
            details: list[str] = []
            for error in errors:
                if isinstance(error, dict):
                    code = error.get("code")
                    message = error.get("message") or error
                    details.append(f"{code}: {message}" if code is not None else str(message))
                else:
                    details.append(str(error))
            detail = "; ".join(details)
        else:
            detail = str(body.get("error") or body)
        return (
            "Cloudflare Workers AI image generation failed with status "
            f"{status_code}: {detail}"
        )

    def _is_content_safety_rejection(
        self, *, response: httpx.Response, message: str
    ) -> bool:
        if response.status_code != 400:
            return False
        lowered = message.lower()
        return "3030" in message or "output has been flagged" in lowered
