import json
import unittest

import httpx

from app.core.exceptions import IntegrationError
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


class GroqClientErrorTests(unittest.TestCase):
    def make_client(
        self,
        handler,
        *,
        api_key: str | None = "groq-test-key",
    ) -> GroqClient:
        return GroqClient(
            api_key=api_key,
            base_url="https://api.groq.test/openai/v1",
            timeout_seconds=30,
            transport=httpx.MockTransport(handler),
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
        client = self.make_client(
            lambda request: httpx.Response(
                401,
                json={"error": {"message": "invalid API key"}},
            )
        )
        self.assert_integration_error(
            lambda: client.generate_text(
                model="llama-3.3-70b-versatile",
                prompt="story",
            ),
            "Groq text API failed with status 401: invalid API key",
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
