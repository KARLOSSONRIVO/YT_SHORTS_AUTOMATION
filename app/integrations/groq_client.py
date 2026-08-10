from __future__ import annotations

import httpx


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

    def generate_text(
        self,
        *,
        model: str,
        prompt: str,
        max_new_tokens: int = 1600,
        temperature: float = 0.7,
    ) -> str:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_completion_tokens": max_new_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key or ''}",
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
        body = response.json()
        return str(body["choices"][0]["message"]["content"])
