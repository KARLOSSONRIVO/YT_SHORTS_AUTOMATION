import json
import unittest

import httpx


class GroqClientRequestTests(unittest.TestCase):
    def test_generate_text_sends_chat_completion_request_and_returns_content(self) -> None:
        try:
            from app.integrations.groq_client import GroqClient
        except ModuleNotFoundError:
            self.fail("GroqClient is not implemented")

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


if __name__ == "__main__":
    unittest.main()
