import json
import unittest

import httpx

from app.core.exceptions import IntegrationError, ProviderRateLimitError
from app.integrations.groq_client import GroqClient


class GroqClientRequestTests(unittest.TestCase):
    def test_generate_text_sends_chat_completion_request_and_returns_content(self) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["authorization"] = request.headers.get("Authorization")
            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": '{"title":"A strong hook"}'}}
                    ]
                },
            )

        client = GroqClient(
            api_key="groq-test-key",
            base_url="https://api.groq.test/openai/v1/",
            timeout_seconds=30,
            transport=httpx.MockTransport(handler),
        )

        result = client.generate_text(
            model="llama-3.3-70b-versatile",
            prompt="Return a story as JSON.",
            max_new_tokens=2200,
            temperature=0.75,
        )

        self.assertEqual(result, '{"title":"A strong hook"}')
        self.assertEqual(
            captured["url"],
            "https://api.groq.test/openai/v1/chat/completions",
        )
        self.assertEqual(captured["authorization"], "Bearer groq-test-key")
        self.assertEqual(
            captured["payload"],
            {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "user", "content": "Return a story as JSON."}
                ],
                "temperature": 0.75,
                "max_completion_tokens": 2200,
                "response_format": {"type": "json_object"},
            },
        )

    def test_qwen_script_generation_disables_reasoning_mode(self) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"title":"Qwen"}'}}]},
            )

        client = GroqClient(
            api_key="groq-test-key",
            base_url="https://api.groq.test/openai/v1",
            timeout_seconds=30,
            transport=httpx.MockTransport(handler),
        )

        client.generate_text(
            model="qwen/qwen3.6-27b",
            prompt="Return a story as JSON.",
        )

        payload = captured["payload"]
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload.get("reasoning_effort"), "none")


class GroqClientErrorTests(unittest.TestCase):
    def make_client(
        self,
        handler,
        *,
        api_key: str | None = "groq-test-key",
        fallback_model: str | None = None,
    ) -> GroqClient:
        return GroqClient(
            api_key=api_key,
            base_url="https://api.groq.test/openai/v1",
            timeout_seconds=30,
            transport=httpx.MockTransport(handler),
            fallback_model=fallback_model,
        )

    def assert_integration_error(self, action, expected_message: str) -> None:
        try:
            action()
        except Exception as exc:
            self.assertIsInstance(exc, IntegrationError)
            self.assertIn(expected_message, str(exc))
        else:
            self.fail("Expected IntegrationError")

    def test_is_configured_requires_an_api_key(self) -> None:
        configured = self.make_client(
            lambda request: httpx.Response(200),
            api_key="key",
        )
        unconfigured = self.make_client(
            lambda request: httpx.Response(200),
            api_key=None,
        )

        self.assertTrue(
            hasattr(configured, "is_configured"),
            "GroqClient.is_configured is not implemented",
        )
        self.assertTrue(configured.is_configured())
        self.assertFalse(unconfigured.is_configured())

    def test_generate_text_requires_api_key(self) -> None:
        client = self.make_client(
            lambda request: httpx.Response(200),
            api_key=None,
        )
        self.assert_integration_error(
            lambda: client.generate_text(
                model="llama-3.3-70b-versatile",
                prompt="story",
            ),
            "GROQ_API_KEY is required",
        )

    def test_generate_text_requires_model(self) -> None:
        client = self.make_client(lambda request: httpx.Response(200))
        self.assert_integration_error(
            lambda: client.generate_text(model="", prompt="story"),
            "GROQ_MODEL is required",
        )

    def test_generate_text_reports_http_errors(self) -> None:
        requested_models: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requested_models.append(json.loads(request.content)["model"])
            return httpx.Response(
                401,
                json={"error": {"message": "invalid API key"}},
            )

        client = self.make_client(
            handler,
            fallback_model="openai/gpt-oss-20b",
        )
        self.assert_integration_error(
            lambda: client.generate_text(
                model="llama-3.3-70b-versatile",
                prompt="story",
            ),
            "Groq text API failed with status 401: invalid API key",
        )
        self.assertEqual(requested_models, ["llama-3.3-70b-versatile"])

    def test_generate_text_falls_back_once_after_rate_limit(self) -> None:
        requested_payloads: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            requested_payloads.append(payload)
            if len(requested_payloads) == 1:
                return httpx.Response(
                    429,
                    json={"error": {"message": "daily token limit reached"}},
                )
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"title":"Fallback"}'}}]},
            )

        client = self.make_client(
            handler,
            fallback_model="openai/gpt-oss-20b",
        )

        result = client.generate_text(
            model="qwen/qwen3.6-27b",
            prompt="Return a story as JSON.",
            max_new_tokens=2200,
            temperature=0.75,
        )

        self.assertEqual(result, '{"title":"Fallback"}')
        self.assertEqual(
            [payload["model"] for payload in requested_payloads],
            ["qwen/qwen3.6-27b", "openai/gpt-oss-20b"],
        )
        self.assertEqual(requested_payloads[0].get("reasoning_effort"), "none")
        self.assertNotIn("reasoning_effort", requested_payloads[1])
        comparable_primary = {
            key: value
            for key, value in requested_payloads[0].items()
            if key not in {"model", "reasoning_effort"}
        }
        comparable_fallback = {
            key: value
            for key, value in requested_payloads[1].items()
            if key != "model"
        }
        self.assertEqual(comparable_primary, comparable_fallback)

    def test_generate_text_raises_typed_error_when_primary_and_fallback_are_rate_limited(self) -> None:
        requested_models: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requested_models.append(json.loads(request.content)["model"])
            return httpx.Response(
                429,
                json={"error": {"message": "token quota reached"}},
            )

        client = self.make_client(
            handler,
            fallback_model="openai/gpt-oss-20b",
        )

        with self.assertRaises(ProviderRateLimitError) as caught:
            client.generate_text(
                model="qwen/qwen3.6-27b",
                prompt="Return a story as JSON.",
            )

        self.assertEqual(
            requested_models,
            ["qwen/qwen3.6-27b", "openai/gpt-oss-20b"],
        )
        self.assertEqual(caught.exception.code, "provider_rate_limit")
        self.assertIn("status 429", str(caught.exception))

    def test_generate_text_preserves_primary_rate_limit_when_fallback_rejects_json(self) -> None:
        requested_models: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requested_models.append(json.loads(request.content)["model"])
            if len(requested_models) == 1:
                return httpx.Response(
                    429,
                    headers={
                        "retry-after": "42",
                        "x-ratelimit-remaining-tokens": "0",
                        "x-ratelimit-reset-tokens": "41.5s",
                    },
                    json={"error": {"message": "TPM limit reached"}},
                )
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "Failed to validate JSON. Please adjust your prompt."
                    }
                },
            )

        client = self.make_client(
            handler,
            fallback_model="openai/gpt-oss-20b",
        )

        with self.assertRaises(ProviderRateLimitError) as caught:
            client.generate_text(
                model="qwen/qwen3.6-27b",
                prompt="Return a story as JSON.",
            )

        self.assertEqual(
            requested_models,
            ["qwen/qwen3.6-27b", "openai/gpt-oss-20b"],
        )
        self.assertIn("status 429: TPM limit reached", str(caught.exception))
        self.assertEqual(caught.exception.response_headers["retry-after"], "42")
        self.assertEqual(
            caught.exception.response_headers["x-ratelimit-remaining-tokens"],
            "0",
        )
        self.assertEqual(
            caught.exception.response_headers["x-ratelimit-reset-tokens"],
            "41.5s",
        )

    def test_generate_text_rejects_non_json_response(self) -> None:
        client = self.make_client(
            lambda request: httpx.Response(200, text="not-json")
        )
        self.assert_integration_error(
            lambda: client.generate_text(
                model="llama-3.3-70b-versatile",
                prompt="story",
            ),
            "Groq returned a non-JSON response",
        )

    def test_generate_text_requires_assistant_content(self) -> None:
        client = self.make_client(
            lambda request: httpx.Response(200, json={"choices": []})
        )
        self.assert_integration_error(
            lambda: client.generate_text(
                model="llama-3.3-70b-versatile",
                prompt="story",
            ),
            "returned no text output",
        )


if __name__ == "__main__":
    unittest.main()
