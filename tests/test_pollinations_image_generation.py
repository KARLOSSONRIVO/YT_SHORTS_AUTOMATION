import base64
import json
import unittest

import httpx

from app.core.exceptions import IntegrationError, PaymentRequiredError, ProviderRateLimitError


class PollinationsImageClientTests(unittest.TestCase):
    def client_type(self):
        try:
            from app.integrations.pollinations_client import PollinationsClient
        except ImportError as exc:
            self.fail(f"Pollinations image client is not implemented: {exc}")
        return PollinationsClient

    def make_client(self, handler):
        return self.client_type()(
            api_key="pollinations-test-key",
            base_url="https://gen.pollinations.test/v1",
            timeout_seconds=30,
            transport=httpx.MockTransport(handler),
        )

    def test_generate_image_posts_vertical_flux_request_and_decodes_base64(self) -> None:
        encoded = base64.b64encode(b"pollinations-flux-image").decode("ascii")

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(
                str(request.url),
                "https://gen.pollinations.test/v1/images/generations",
            )
            self.assertEqual(
                request.headers["Authorization"],
                "Bearer pollinations-test-key",
            )
            payload = json.loads(request.content)
            self.assertEqual(payload["model"], "flux")
            self.assertEqual(payload["size"], "768x1024")
            self.assertEqual(payload["response_format"], "b64_json")
            self.assertEqual(payload["n"], 1)
            self.assertEqual(
                payload["prompt"],
                "cinematic scene\nAvoid: text, watermark",
            )
            return httpx.Response(200, json={"data": [{"b64_json": encoded}]})

        result = self.make_client(handler).generate_image(
            model="flux",
            prompt="cinematic scene",
            negative_prompt="text, watermark",
            width=768,
            height=1024,
        )

        self.assertEqual(result["image_bytes"], b"pollinations-flux-image")
        self.assertEqual(result["mime_type"], "image/png")
        self.assertEqual(result["provider"], "pollinations")

    def test_generate_image_maps_429_to_provider_rate_limit(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429,
                json={"error": {"message": "rate limit exceeded"}},
            )

        with self.assertRaises(ProviderRateLimitError) as caught:
            self.make_client(handler).generate_image(
                model="flux",
                prompt="scene",
            )

        self.assertIn("status 429", str(caught.exception))
        self.assertIn("rate limit exceeded", str(caught.exception))

    def test_generate_image_maps_exhausted_pollen_to_payment_required(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                402,
                json={"error": {"message": "Pollen balance exhausted"}},
            )

        with self.assertRaises(PaymentRequiredError) as caught:
            self.make_client(handler).generate_image(
                model="flux",
                prompt="scene",
            )

        self.assertIn("status 402", str(caught.exception))
        self.assertIn("Pollen balance exhausted", str(caught.exception))

    def test_generate_image_rejects_invalid_base64_response(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": [{"b64_json": "not-base64!"}]})

        with self.assertRaises(IntegrationError) as caught:
            self.make_client(handler).generate_image(model="flux", prompt="scene")

        self.assertIn("valid image data", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
