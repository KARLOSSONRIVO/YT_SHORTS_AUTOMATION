import unittest
from unittest.mock import patch

import httpx

from app.core.exceptions import IntegrationError, ProviderRateLimitError
from app.integrations.gemini_client import GeminiClient


class GeminiClientErrorTests(unittest.TestCase):
    def make_client(self) -> GeminiClient:
        return GeminiClient(
            api_key="gemini-test-key",
            base_url="https://generativelanguage.test/v1beta/interactions",
            timeout_seconds=30,
        )

    def test_generate_image_raises_typed_error_for_rate_limit(self) -> None:
        response = httpx.Response(
            429,
            json={
                "error": {
                    "code": "rate_limit_exceeded",
                    "message": "image quota reached",
                }
            },
        )
        with patch("app.integrations.gemini_client.httpx.Client") as client_type:
            client_type.return_value.__enter__.return_value.post.return_value = response
            with self.assertRaises(ProviderRateLimitError) as caught:
                self.make_client().generate_image(
                    model="gemini-image-model",
                    prompt="scene",
                )

        self.assertEqual(caught.exception.code, "provider_rate_limit")
        self.assertIn("status 429", str(caught.exception))
        self.assertIn("image quota reached", str(caught.exception))

    def test_generate_image_keeps_non_rate_limit_as_integration_error(self) -> None:
        response = httpx.Response(
            403,
            json={
                "error": {
                    "code": "permission_denied",
                    "message": "permission denied",
                }
            },
        )
        with patch("app.integrations.gemini_client.httpx.Client") as client_type:
            client_type.return_value.__enter__.return_value.post.return_value = response
            with self.assertRaises(IntegrationError) as caught:
                self.make_client().generate_image(
                    model="gemini-image-model",
                    prompt="scene",
                )

        self.assertNotIsInstance(caught.exception, ProviderRateLimitError)
        self.assertIn("status 403", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
