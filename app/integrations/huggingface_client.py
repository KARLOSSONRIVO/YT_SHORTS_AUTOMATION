import json
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
from huggingface_hub import InferenceClient

from app.core.exceptions import IntegrationError, PaymentRequiredError


class HuggingFaceClient:
    def __init__(
        self,
        *,
        token: str | None,
        inference_base_url: str,
        timeout_seconds: float,
        router_base_url: str = "https://router.huggingface.co/v1",
        max_retries: int = 2,
    ) -> None:
        self.token = token
        self.inference_base_url = inference_base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.router_base_url = router_base_url.rstrip("/")
        self.max_retries = max_retries
        self.inference_client = InferenceClient(
            provider="auto",
            api_key=token,
            timeout=timeout_seconds,
        )

    def is_configured(self) -> bool:
        return bool(self.token)

    def generate_text(
        self,
        *,
        model: str,
        prompt: str,
        max_new_tokens: int = 1600,
        temperature: float = 0.7,
    ) -> str:
        if not self.token:
            raise IntegrationError("PY_WORKER_HF_TOKEN is required for Hugging Face chat inference.")

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "stream": False,
        }
        response = self._post_router_json("/chat/completions", payload)
        return self._extract_generated_text(response)

    def text_to_image(self, *, model: str, prompt: str) -> bytes:
        if not self.token:
            raise IntegrationError("PY_WORKER_HF_TOKEN is required for Hugging Face image inference.")

        try:
            image = self.inference_client.text_to_image(prompt=prompt, model=model)
        except Exception as exc:
            if _is_payment_required(exc):
                raise PaymentRequiredError(
                    f"Hugging Face image model requires a paid plan (HTTP 402): {exc}"
                ) from exc
            raise IntegrationError(f"Hugging Face text-to-image failed: {exc}") from exc

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def text_to_speech(self, *, model: str, text: str, voice: str | None = None) -> bytes:
        if not self.token:
            raise IntegrationError("PY_WORKER_HF_TOKEN is required for Hugging Face TTS inference.")

        try:
            return self.inference_client.text_to_speech(
                text,
                model=model,
                extra_body={"voice": voice} if voice else None,
            )
        except Exception as exc:
            raise IntegrationError(f"Hugging Face text-to-speech failed: {exc}") from exc

    def transcribe_audio(self, *, model: str, audio_path: str) -> dict[str, Any]:
        if not self.token:
            raise IntegrationError("PY_WORKER_HF_TOKEN is required for Hugging Face transcription.")

        try:
            result = self.inference_client.automatic_speech_recognition(Path(audio_path), model=model)
        except Exception as exc:
            raise IntegrationError(f"Hugging Face speech-to-text failed: {exc}") from exc

        return self._output_to_dict(result)

    def _post_router_json(self, path: str, payload: dict[str, Any]) -> Any:
        if not self.token:
            raise IntegrationError("PY_WORKER_HF_TOKEN is required for Hugging Face inference.")

        url = f"{self.router_base_url}{path}"
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(url, headers=self._headers(), json=payload)

        if response.status_code >= 400:
            raise IntegrationError(self._error_message(response))

        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise IntegrationError("Hugging Face router returned a non-JSON response.") from exc

    def _post_json(self, model: str, payload: dict[str, Any]) -> Any:
        if not self.token:
            raise IntegrationError("PY_WORKER_HF_TOKEN is required for Hugging Face inference.")

        response = self._request(model, payload)
        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise IntegrationError("Hugging Face returned a non-JSON text response.") from exc

    def _post_binary(self, model: str, payload: dict[str, Any], expected_prefix: str) -> bytes:
        if not self.token:
            raise IntegrationError("PY_WORKER_HF_TOKEN is required for Hugging Face inference.")

        response = self._request(model, payload)
        content_type = response.headers.get("content-type", "")
        if content_type.startswith("application/json"):
            try:
                body = response.json()
            except json.JSONDecodeError as exc:
                raise IntegrationError("Hugging Face returned invalid JSON instead of binary content.") from exc
            raise IntegrationError(str(body.get("error") or body))

        if expected_prefix and not content_type.startswith(expected_prefix):
            raise IntegrationError(
                f"Hugging Face returned '{content_type or 'unknown'}' instead of {expected_prefix} content."
            )

        return response.content

    def _request(self, model: str, payload: dict[str, Any]) -> httpx.Response:
        last_response: httpx.Response | None = None
        for attempt in range(self.max_retries + 1):
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(self._model_url(model), headers=self._headers(), json=payload)

            if response.status_code < 400:
                return response

            last_response = response
            if response.status_code not in {429, 503} or attempt >= self.max_retries:
                break

            wait_seconds = self._retry_delay(response, attempt)
            time.sleep(wait_seconds)

        if last_response is None:
            raise IntegrationError("Hugging Face request failed before receiving a response.")
        raise IntegrationError(self._error_message(last_response))

    def _model_url(self, model: str) -> str:
        return f"{self.inference_base_url}/{model.strip('/')}"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float:
        try:
            body = response.json()
        except json.JSONDecodeError:
            body = {}

        estimated_time = body.get("estimated_time") if isinstance(body, dict) else None
        if isinstance(estimated_time, int | float):
            return min(float(estimated_time), 20.0)
        return min(2.0 * (attempt + 1), 8.0)

    def _error_message(self, response: httpx.Response) -> str:
        try:
            body = response.json()
        except json.JSONDecodeError:
            body = response.text

        if isinstance(body, dict) and body.get("error"):
            return f"Hugging Face inference failed: {body['error']}"
        return f"Hugging Face inference failed with status {response.status_code}: {body}"

    def _extract_generated_text(self, response: Any) -> str:
        if isinstance(response, list) and response:
            first = response[0]
            if isinstance(first, dict) and isinstance(first.get("generated_text"), str):
                return first["generated_text"]

        if isinstance(response, dict):
            if isinstance(response.get("generated_text"), str):
                return response["generated_text"]
            if isinstance(response.get("text"), str):
                return response["text"]
            choices = response.get("choices")
            if isinstance(choices, list) and choices:
                first_choice = choices[0]
                if isinstance(first_choice, dict):
                    message = first_choice.get("message")
                    if isinstance(message, dict) and isinstance(message.get("content"), str):
                        return message["content"]
                    if isinstance(first_choice.get("text"), str):
                        return first_choice["text"]

        raise IntegrationError(f"Unexpected Hugging Face text-generation response: {response}")

    def _output_to_dict(self, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value

        if hasattr(value, "model_dump"):
            dumped = value.model_dump()
            if isinstance(dumped, dict):
                return dumped

        text = getattr(value, "text", None)
        chunks = getattr(value, "chunks", None)
        result: dict[str, Any] = {}
        if isinstance(text, str):
            result["text"] = text
        if chunks is not None:
            result["chunks"] = chunks

        if result:
            return result

        if isinstance(value, str):
            return {"text": value}

        raise IntegrationError(f"Unexpected Hugging Face speech-to-text response: {value}")


def _is_payment_required(exc: Exception) -> bool:
    """Return True if *exc* was caused by an HTTP 402 Payment Required response."""
    # huggingface_hub raises HfHubHTTPError with a .response attribute
    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", None)
        if status == 402:
            return True
    # Also check stringified message as a safety net
    if "402" in str(exc):
        return True
    return False
