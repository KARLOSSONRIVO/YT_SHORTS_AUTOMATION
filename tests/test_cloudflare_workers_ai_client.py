import base64
import json
import unittest

import httpx

from app.core.exceptions import (
    ContentSafetyError,
    IntegrationError,
    ProviderRateLimitError,
)
from app.integrations.cloudflare_workers_ai_client import CloudflareWorkersAIClient


class CloudflareWorkersAIClientTests(unittest.TestCase):
    def make_client(self, handler) -> CloudflareWorkersAIClient:
        return CloudflareWorkersAIClient(
            account_id="account-123",
            api_token="cloudflare-test-token",
            base_url="https://api.cloudflare.test/client/v4/accounts",
            timeout_seconds=30,
            transport=httpx.MockTransport(handler),
        )

    def test_generate_image_posts_vertical_flux_klein_9b_multipart_payload(self) -> None:
        encoded = base64.b64encode(b"flux-image-bytes").decode("ascii")

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(
                str(request.url),
                "https://api.cloudflare.test/client/v4/accounts/account-123/ai/run/"
                "@cf/black-forest-labs/flux-2-klein-9b",
            )
            self.assertEqual(
                request.headers["Authorization"],
                "Bearer cloudflare-test-token",
            )
            self.assertTrue(
                request.headers["content-type"].startswith(
                    "multipart/form-data; boundary="
                )
            )
            body = request.content
            self.assertIn(b'name="prompt"', body)
            self.assertIn(b"cinematic scene", body)
            self.assertIn(b"Avoid: text, watermark", body)
            self.assertIn(b'name="width"', body)
            self.assertIn(b"1024", body)
            self.assertIn(b'name="height"', body)
            self.assertIn(b"1792", body)
            self.assertIn(b'name="guidance"', body)
            self.assertIn(b"7.5", body)
            self.assertNotIn(b'name="num_steps"', body)
            self.assertNotIn(b'name="negative_prompt"', body)
            return httpx.Response(
                200,
                json={
                    "result": {"image": encoded, "mime_type": "image/png"},
                    "success": True,
                    "errors": [],
                    "messages": [],
                },
            )

        result = self.make_client(handler).generate_image(
            model="@cf/black-forest-labs/flux-2-klein-9b",
            prompt="cinematic scene",
            negative_prompt="text, watermark",
            width=1024,
            height=1792,
            num_steps=8,
            guidance=7.5,
        )

        self.assertEqual(result["image_bytes"], b"flux-image-bytes")
        self.assertEqual(result["mime_type"], "image/png")
        self.assertEqual(result["provider"], "cloudflare_workers_ai")

    def test_generate_image_posts_quality_steps_for_flux_dev(self) -> None:
        encoded = base64.b64encode(b"flux-dev-image-bytes").decode("ascii")

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(
                str(request.url),
                "https://api.cloudflare.test/client/v4/accounts/account-123/ai/run/"
                "@cf/black-forest-labs/flux-2-dev",
            )
            self.assertTrue(
                request.headers["content-type"].startswith(
                    "multipart/form-data; boundary="
                )
            )
            body = request.content
            self.assertIn(b'name="prompt"', body)
            self.assertIn(b"Avoid: text, watermark", body)
            self.assertIn(b'name="steps"', body)
            self.assertIn(b"25", body)
            self.assertIn(b'name="width"', body)
            self.assertIn(b"768", body)
            self.assertIn(b'name="height"', body)
            self.assertIn(b"1024", body)
            self.assertNotIn(b'name="num_steps"', body)
            self.assertNotIn(b'name="negative_prompt"', body)
            return httpx.Response(
                200,
                json={
                    "result": {"image": encoded, "mime_type": "image/png"},
                    "success": True,
                    "errors": [],
                    "messages": [],
                },
            )

        result = self.make_client(handler).generate_image(
            model="@cf/black-forest-labs/flux-2-dev",
            prompt="cinematic tennis scene",
            negative_prompt="text, watermark",
            width=768,
            height=1024,
            num_steps=25,
            guidance=7.5,
        )

        self.assertEqual(result["image_bytes"], b"flux-dev-image-bytes")

    def test_generate_image_accepts_json_base64_envelope(self) -> None:
        encoded = base64.b64encode(b"image-from-json").decode("ascii")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "result": {"image": encoded, "mime_type": "image/png"},
                    "success": True,
                    "errors": [],
                    "messages": [],
                },
            )

        result = self.make_client(handler).generate_image(
            model="@cf/bytedance/stable-diffusion-xl-lightning",
            prompt="scene",
        )

        self.assertEqual(result["image_bytes"], b"image-from-json")
        self.assertEqual(result["mime_type"], "image/png")

    def test_generate_image_maps_cloudflare_429_to_provider_rate_limit(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429,
                json={
                    "success": False,
                    "errors": [
                        {
                            "code": 3036,
                            "message": "daily free allocation exhausted",
                        }
                    ],
                },
            )

        with self.assertRaises(ProviderRateLimitError) as caught:
            self.make_client(handler).generate_image(
                model="@cf/bytedance/stable-diffusion-xl-lightning",
                prompt="scene",
            )

        self.assertEqual(caught.exception.code, "provider_rate_limit")
        self.assertIn("status 429", str(caught.exception))
        self.assertIn("3036", str(caught.exception))
        self.assertIn("daily free allocation exhausted", str(caught.exception))

    def test_generate_image_logs_safe_cloudflare_error_details(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                json={
                    "success": False,
                    "errors": [
                        {
                            "code": 3003,
                            "message": "request body was rejected",
                        }
                    ],
                },
            )

        model = "@cf/black-forest-labs/flux-2-klein-4b"
        with self.assertLogs(
            "app.integrations.cloudflare_workers_ai_client",
            level="ERROR",
        ) as captured:
            with self.assertRaises(IntegrationError):
                self.make_client(handler).generate_image(
                    model=model,
                    prompt="secret-scene-prompt",
                )

        logged = "\n".join(captured.output)
        self.assertIn(model, logged)
        self.assertIn("status=400", logged)
        self.assertIn("3003: request body was rejected", logged)
        self.assertNotIn("cloudflare-test-token", logged)
        self.assertNotIn("secret-scene-prompt", logged)

    def test_generate_image_maps_cloudflare_content_safety_rejection(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                json={
                    "success": False,
                    "errors": [
                        {
                            "code": 3030,
                            "message": "Your output has been flagged. Please choose another prompt",
                        }
                    ],
                },
            )

        with self.assertRaises(ContentSafetyError) as caught:
            self.make_client(handler).generate_image(
                model="@cf/black-forest-labs/flux-2-klein-4b",
                prompt="unsafe scene prompt",
            )

        self.assertEqual(caught.exception.code, "provider_content_safety")
        self.assertIn("3030", str(caught.exception))

    def test_generate_image_requires_cloudflare_credentials(self) -> None:
        client = CloudflareWorkersAIClient(
            account_id=None,
            api_token=None,
            base_url="https://api.cloudflare.test/client/v4/accounts",
            timeout_seconds=30,
        )

        with self.assertRaises(IntegrationError) as caught:
            client.generate_image(
                model="@cf/bytedance/stable-diffusion-xl-lightning",
                prompt="scene",
            )

        self.assertIn("CLOUDFLARE_ACCOUNT_ID", str(caught.exception))
        self.assertIn("CLOUDFLARE_API_TOKEN", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
