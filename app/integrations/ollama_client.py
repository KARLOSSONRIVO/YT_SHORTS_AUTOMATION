import json
import re

import httpx

from app.core.exceptions import IntegrationError


class OllamaClient:
    """Client for a locally running Ollama instance.

    Uses the native Ollama ``/api/chat`` endpoint so that thinking/reasoning
    models (e.g. qwen3.5, qwen3) can be controlled with ``think: false``.
    The OpenAI-compatible ``/v1/chat/completions`` endpoint ignores that flag
    on many Ollama versions, which causes empty ``message.content`` responses.
    """

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float = 300.0,
        max_retries: int = 2,
    ) -> None:
        # Accept URLs like "http://host:11434/v1" or "http://host:11434"
        # and normalise to the bare origin so we can hit /api/chat.
        cleaned = base_url.rstrip("/")
        cleaned = re.sub(r"/v\d+$", "", cleaned)  # strip trailing /v1, /v2, etc.
        self.base_url = cleaned
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def generate_text(
        self,
        *,
        model: str,
        prompt: str,
        max_new_tokens: int = 1600,
        temperature: float = 0.7,
    ) -> str:
        """Call Ollama's native /api/chat endpoint with thinking disabled."""

        payload: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": temperature,
            },
        }

        url = f"{self.base_url}/api/chat"
        last_exc: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    response = client.post(url, json=payload)
            except httpx.ConnectError as exc:
                raise IntegrationError(
                    f"Cannot connect to Ollama at {self.base_url}. "
                    "Ensure Ollama is running and PY_WORKER_OLLAMA_BASE_URL is correct."
                ) from exc
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                continue

            if response.status_code >= 400:
                raise IntegrationError(
                    f"Ollama request failed with status {response.status_code}: {response.text}"
                )

            try:
                data = response.json()
            except json.JSONDecodeError as exc:
                raise IntegrationError("Ollama returned a non-JSON response.") from exc

            # Native Ollama /api/chat response shape:
            #   {"message": {"role": "assistant", "content": "..."}, "done": true, ...}
            message = data.get("message") or {}
            content = message.get("content", "")
            if isinstance(content, str) and content:
                return content

            raise IntegrationError(f"Unexpected Ollama response structure: {data}")

        raise IntegrationError(
            f"Ollama request timed out after {self.max_retries + 1} attempts."
        ) from last_exc
