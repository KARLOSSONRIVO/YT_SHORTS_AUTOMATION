from __future__ import annotations

import json
from typing import Any

import httpx

from app.core.exceptions import IntegrationError


class GroqClient:
    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str,
        timeout_seconds: float | None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = (
            timeout_seconds if timeout_seconds and timeout_seconds > 0 else None
        )
        self.transport = transport

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
            raise IntegrationError(
                "GROQ_API_KEY is required for Groq story generation."
            )
        if not model:
            raise IntegrationError("GROQ_MODEL is required for Groq story generation.")

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_completion_tokens": max_new_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(
            timeout=self.timeout_seconds,
            transport=self.transport,
        ) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )

        if response.status_code >= 400:
            raise IntegrationError(self._error_message(response))

        try:
            body = response.json()
        except json.JSONDecodeError as exc:
            raise IntegrationError(
                "Groq text API failed: Groq returned a non-JSON response."
            ) from exc

        content = self._extract_content(body)
        if not content:
            raise IntegrationError(f"Groq text API returned no text output: {body}")
        return content

    def _extract_content(self, body: Any) -> str | None:
        if not isinstance(body, dict):
            return None
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        first = choices[0]
        if not isinstance(first, dict):
            return None
        message = first.get("message")
        if not isinstance(message, dict):
            return None
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
        return None

    def _error_message(self, response: httpx.Response) -> str:
        try:
            body = response.json()
        except json.JSONDecodeError:
            body = response.text

        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                message = error.get("message") or error
                return (
                    f"Groq text API failed with status {response.status_code}: "
                    f"{message}"
                )
            if error:
                return (
                    f"Groq text API failed with status {response.status_code}: "
                    f"{error}"
                )
        return f"Groq text API failed with status {response.status_code}: {body}"
