import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.api import deps
from app.core.config import Settings


class GroqSettingsTests(unittest.TestCase):
    def test_settings_load_bare_groq_key_and_default_story_model(self) -> None:
        with patch.dict(os.environ, {"GROQ_API_KEY": "groq-env-key"}, clear=True):
            settings = Settings(_env_file=None)

        self.assertTrue(
            hasattr(settings, "groq_api_key"),
            "Groq settings are not implemented",
        )
        self.assertEqual(settings.groq_api_key, "groq-env-key")
        self.assertEqual(settings.groq_model, "qwen/qwen3.6-27b")
        self.assertEqual(
            settings.groq_fallback_model,
            "llama-3.1-8b-instant",
        )
        self.assertEqual(
            settings.groq_base_url,
            "https://api.groq.com/openai/v1",
        )

    def test_empty_prefixed_key_does_not_mask_configured_gemini_key(self) -> None:
        with patch.dict(
            os.environ,
            {
                "PY_WORKER_GEMINI_API_KEY": "",
                "GEMINI_API_KEY": "gemini-env-key",
            },
            clear=True,
        ):
            settings = Settings(_env_file=None)

        self.assertEqual(settings.gemini_api_key, "gemini-env-key")

    def test_settings_load_cloudflare_image_provider_defaults(self) -> None:
        with patch.dict(
            os.environ,
            {
                "CLOUDFLARE_ACCOUNT_ID": "account-123",
                "CLOUDFLARE_API_TOKEN": "cloudflare-token",
            },
            clear=True,
        ):
            settings = Settings(_env_file=None)

        self.assertEqual(settings.cloudflare_account_id, "account-123")
        self.assertEqual(settings.cloudflare_api_token, "cloudflare-token")
        self.assertEqual(
            settings.cloudflare_image_model,
            "@cf/black-forest-labs/flux-2-klein-4b",
        )
        self.assertEqual(settings.cloudflare_image_width, 1024)
        self.assertEqual(settings.cloudflare_image_height, 1792)
        self.assertEqual(settings.cloudflare_image_num_steps, 8)


class ProviderRoutingTests(unittest.TestCase):
    def tearDown(self) -> None:
        deps.get_groq_client.cache_clear()
        deps.get_llm_service.cache_clear()
        deps.get_cloudflare_workers_ai_client.cache_clear()
        deps.get_cloudflare_image_generation_service.cache_clear()
        deps.get_tts_service.cache_clear()

    def test_llm_service_receives_groq_client_and_model(self) -> None:
        settings = SimpleNamespace(
            groq_model="qwen/qwen3.6-27b",
            gemini_model="gemini-story-model",
            allow_placeholder_generation=False,
        )
        groq_client = object()
        gemini_client = object()
        deps.get_llm_service.cache_clear()

        with (
            patch.object(deps, "get_settings", return_value=settings),
            patch.object(
                deps,
                "get_groq_client",
                return_value=groq_client,
                create=True,
            ),
            patch.object(
                deps,
                "get_gemini_client",
                return_value=gemini_client,
            ),
        ):
            service = deps.get_llm_service()

        self.assertIs(service.llm_client, groq_client)
        self.assertEqual(service.model, "qwen/qwen3.6-27b")

    def test_groq_client_receives_configured_fallback_model(self) -> None:
        settings = SimpleNamespace(
            groq_api_key="groq-test-key",
            groq_base_url="https://api.groq.test/openai/v1",
            groq_fallback_model="llama-3.1-8b-instant",
            ai_timeout_seconds=30,
        )
        deps.get_groq_client.cache_clear()

        with (
            patch.object(deps, "get_settings", return_value=settings),
            patch.object(deps, "GroqClient") as groq_client_class,
        ):
            deps.get_groq_client()

        groq_client_class.assert_called_once_with(
            api_key="groq-test-key",
            base_url="https://api.groq.test/openai/v1",
            timeout_seconds=30,
            fallback_model="llama-3.1-8b-instant",
        )

    def test_image_generation_service_receives_cloudflare_client_and_settings(self) -> None:
        settings = SimpleNamespace(
            cloudflare_image_model="@cf/black-forest-labs/flux-2-klein-4b",
            cloudflare_image_width=1024,
            cloudflare_image_height=1792,
            cloudflare_image_num_steps=8,
            cloudflare_image_guidance=7.5,
        )
        cloudflare_client = object()
        deps.get_cloudflare_image_generation_service.cache_clear()

        with (
            patch.object(deps, "get_settings", return_value=settings),
            patch.object(
                deps,
                "get_cloudflare_workers_ai_client",
                return_value=cloudflare_client,
            ),
        ):
            service = deps.get_cloudflare_image_generation_service()

        self.assertIs(service.cloudflare_client, cloudflare_client)
        self.assertEqual(
            service.model,
            "@cf/black-forest-labs/flux-2-klein-4b",
        )
        self.assertEqual(service.width, 1024)
        self.assertEqual(service.height, 1792)
        self.assertEqual(service.num_steps, 8)
        self.assertEqual(service.guidance, 7.5)

    def test_tts_service_still_receives_gemini_client(self) -> None:
        settings = SimpleNamespace(
            output_dir="outputs",
            gemini_tts_model="gemini-tts-model",
            allow_placeholder_generation=False,
        )
        gemini_client = object()
        ffmpeg_client = object()
        deps.get_tts_service.cache_clear()

        with (
            patch.object(deps, "get_settings", return_value=settings),
            patch.object(
                deps,
                "get_gemini_client",
                return_value=gemini_client,
            ),
            patch.object(
                deps,
                "get_ffmpeg_client",
                return_value=ffmpeg_client,
            ),
        ):
            service = deps.get_tts_service()

        self.assertIs(service.gemini_client, gemini_client)
        self.assertIs(service.ffmpeg_client, ffmpeg_client)
        self.assertEqual(service.model, "gemini-tts-model")


if __name__ == "__main__":
    unittest.main()
