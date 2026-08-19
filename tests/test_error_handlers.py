import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.error_handlers import register_exception_handlers
from app.core.exceptions import ProviderRateLimitError


class ErrorHandlerTests(unittest.TestCase):
    def test_provider_rate_limit_maps_to_http_429(self) -> None:
        app = FastAPI()
        register_exception_handlers(app)

        @app.get("/rate-limited")
        def rate_limited() -> None:
            raise ProviderRateLimitError("Groq quota reached")

        response = TestClient(app).get("/rate-limited")

        self.assertEqual(response.status_code, 429)
        self.assertEqual(
            response.json(),
            {
                "error": {
                    "code": "provider_rate_limit",
                    "message": "Groq quota reached",
                }
            },
        )

    def test_provider_rate_limit_forwards_safe_retry_headers(self) -> None:
        app = FastAPI()
        register_exception_handlers(app)

        @app.get("/rate-limited-with-headers")
        def rate_limited() -> None:
            raise ProviderRateLimitError(
                "Groq TPM reached",
                response_headers={
                    "retry-after": "42",
                    "x-ratelimit-limit-tokens": "8000",
                    "x-ratelimit-remaining-tokens": "0",
                    "x-ratelimit-reset-tokens": "41.5s",
                    "authorization": "must-not-forward",
                },
            )

        response = TestClient(app).get("/rate-limited-with-headers")

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["retry-after"], "42")
        self.assertEqual(response.headers["x-ratelimit-limit-tokens"], "8000")
        self.assertEqual(response.headers["x-ratelimit-remaining-tokens"], "0")
        self.assertEqual(response.headers["x-ratelimit-reset-tokens"], "41.5s")
        self.assertNotIn("authorization", response.headers)


if __name__ == "__main__":
    unittest.main()
