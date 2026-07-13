from __future__ import annotations

import base64
import json
from typing import Any

import httpx

from app.core.exceptions import IntegrationError


class GeminiClient:
    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str,
        timeout_seconds: float | None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds if timeout_seconds and timeout_seconds > 0 else None

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def generate_text(
        self,
        *,
        model: str,
        prompt: str,
        max_new_tokens: int = 1600,
        temperature: float = 0.7,
    ) -> str:
        if not self.api_key:
            raise IntegrationError("GEMINI_API_KEY is required for Gemini text generation.")
        if not model:
            raise IntegrationError("GEMINI_MODEL is required for Gemini text generation.")

        payload = {
            "model": model,
            "input": prompt,
            "generation_config": {
                "temperature": temperature,
                "max_output_tokens": max_new_tokens,
            },
        }
        response = self._post_interaction(payload, error_prefix="Gemini text API failed")
        text = self._extract_output_text(response)
        if not text:
            raise IntegrationError(f"Gemini text API returned no text output: {response}")
        return text

    def generate_image(
        self,
        *,
        model: str,
        prompt: str,
        aspect_ratio: str = "9:16",
        image_size: str = "1K",
    ) -> dict[str, Any]:
        if not self.api_key:
            raise IntegrationError("GEMINI_API_KEY is required for Gemini image generation.")
        if not model:
            raise IntegrationError("GEMINI_IMAGE_MODEL is required for Gemini image generation.")

        payload = {
            "model": model,
            "input": prompt,
            "response_format": {
                "type": "image",
                "aspect_ratio": aspect_ratio,
                "image_size": image_size,
            },
        }
        response = self._post_interaction(payload, error_prefix="Gemini image generation failed")
        image = self._extract_output_image(response)
        if image is None:
            raise IntegrationError("Gemini image generation returned no image output.")
        return image

    def generate_speech(self, *, model: str, text: str, voice: str) -> bytes:
        if not self.api_key:
            raise IntegrationError("GEMINI_API_KEY is required for Gemini speech generation.")
        payload = {
            "model": model,
            "input": text,
            "response_format": {"type": "audio"},
            "generation_config": {"speech_config": [{"voice": voice}]},
        }
        response = self._post_interaction(payload, error_prefix="Gemini speech generation failed")
        return self._extract_media_bytes(response, media_type="audio")

    def generate_video(
        self,
        *,
        model: str,
        prompt: str,
        aspect_ratio: str = "9:16",
    ) -> bytes:
        if not self.api_key:
            raise IntegrationError("GEMINI_API_KEY is required for Gemini video generation.")
        payload = {
            "model": model,
            "input": prompt,
            "response_format": {"type": "video", "aspect_ratio": aspect_ratio},
        }
        response = self._post_interaction(payload, error_prefix="Gemini video generation failed")
        return self._extract_media_bytes(response, media_type="video")

    def _post_interaction(self, payload: dict[str, Any], *, error_prefix: str) -> Any:
        headers = {
            "x-goog-api-key": self.api_key or "",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(self.base_url, headers=headers, json=payload)

        if response.status_code >= 400:
            raise IntegrationError(self._error_message(response, prefix=error_prefix))

        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise IntegrationError(f"{error_prefix}: Gemini returned a non-JSON response.") from exc

    def _error_message(self, response: httpx.Response, *, prefix: str) -> str:
        try:
            body = response.json()
        except json.JSONDecodeError:
            body = response.text

        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                message = error.get("message") or error
                return f"{prefix} with status {response.status_code}: {message}"
            if error:
                return f"{prefix} with status {response.status_code}: {error}"
        return f"{prefix} with status {response.status_code}: {body}"

    def _extract_output_text(self, response: Any) -> str | None:
        if isinstance(response, dict):
            output_text = response.get("output_text")
            if isinstance(output_text, str) and output_text.strip():
                return output_text

        text_parts = self._collect_text_parts(response)
        if text_parts:
            return "\n".join(text_parts).strip()
        return None

    def _collect_text_parts(self, value: Any) -> list[str]:
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                parts.extend(self._collect_text_parts(item))
            return parts

        if not isinstance(value, dict):
            return []

        if value.get("type") == "text" and isinstance(value.get("text"), str):
            return [value["text"]]

        collected: list[str] = []
        for key in ("output", "steps", "content", "parts", "messages", "delta"):
            if key in value:
                collected.extend(self._collect_text_parts(value[key]))
        return collected

    def _extract_output_image(self, response: Any) -> dict[str, Any] | None:
        if isinstance(response, list):
            for item in response:
                image = self._extract_output_image(item)
                if image is not None:
                    return image
            return None

        if not isinstance(response, dict):
            return None

        candidate = self._image_from_dict(response)
        if candidate is not None:
            return candidate

        for key in ("output_image", "output_images", "output", "steps", "content", "parts", "messages"):
            if key in response:
                image = self._extract_output_image(response[key])
                if image is not None:
                    return image
        return None

    def _extract_media_bytes(self, response: Any, *, media_type: str) -> bytes:
        if isinstance(response, list):
            for item in response:
                try:
                    return self._extract_media_bytes(item, media_type=media_type)
                except IntegrationError:
                    continue
        elif isinstance(response, dict):
            direct = response.get(f"output_{media_type}")
            if isinstance(direct, dict) and isinstance(direct.get("data"), str):
                return self._decode_media_data(direct["data"], media_type=media_type)
            if response.get("type") == media_type and isinstance(response.get("data"), str):
                return self._decode_media_data(response["data"], media_type=media_type)
            for key in ("output", "steps", "content", "parts", "messages"):
                if key in response:
                    try:
                        return self._extract_media_bytes(response[key], media_type=media_type)
                    except IntegrationError:
                        continue
        raise IntegrationError(f"Gemini response contained no {media_type} output.")

    def _decode_media_data(self, data: str, *, media_type: str) -> bytes:
        try:
            decoded = base64.b64decode(data, validate=True)
        except Exception as exc:
            raise IntegrationError(f"Gemini returned invalid base64 {media_type} data.") from exc
        if not decoded:
            raise IntegrationError(f"Gemini returned empty {media_type} data.")
        return decoded

    def _image_from_dict(self, value: dict[str, Any]) -> dict[str, Any] | None:
        image = value.get("image") if isinstance(value.get("image"), dict) else value
        data = image.get("data") or image.get("bytes") or image.get("base64")
        mime_type = image.get("mime_type") or image.get("mimeType") or image.get("media_type") or "image/png"
        if not isinstance(data, str):
            return None

        try:
            image_bytes = base64.b64decode(data, validate=True)
        except Exception as exc:
            raise IntegrationError("Gemini image generation returned invalid base64 image data.") from exc

        if not image_bytes:
            raise IntegrationError("Gemini image generation returned empty image data.")

        return {
            "image_bytes": image_bytes,
            "mime_type": mime_type,
            "provider": "gemini",
        }
