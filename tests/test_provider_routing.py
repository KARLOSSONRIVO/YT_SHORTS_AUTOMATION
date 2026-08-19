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
            "openai/gpt-oss-20b",
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

    def test_settings_load_pollinations_primary_image_defaults(self) -> None:
        with patch.dict(
            os.environ,
            {"POLLINATIONS_API_KEY": "pollinations-env-key"},
            clear=True,
        ):
            settings = Settings(_env_file=None)

        self.assertEqual(settings.pollinations_api_key, "pollinations-env-key")
        self.assertEqual(settings.pollinations_image_model, "flux")
        self.assertEqual(settings.pollinations_image_width, 768)
        self.assertEqual(settings.pollinations_image_height, 1024)
        self.assertEqual(
            settings.pollinations_base_url,
            "https://gen.pollinations.ai/v1",
        )
        self.assertEqual(settings.pollinations_image_timeout_seconds, 300)
        self.assertEqual(settings.pollinations_tts_model, "elevenlabs")


class ProviderRoutingTests(unittest.TestCase):
    def tearDown(self) -> None:
        deps.get_groq_client.cache_clear()
        deps.get_llm_service.cache_clear()
        for dependency_name in (
            "get_pollinations_client",
            "get_pollinations_image_generation_service",
        ):
            dependency = getattr(deps, dependency_name, None)
            if dependency is not None:
                dependency.cache_clear()
        deps.get_image_service.cache_clear()
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
            groq_fallback_model="openai/gpt-oss-20b",
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
            fallback_model="openai/gpt-oss-20b",
        )

    def test_pollinations_client_receives_configured_key_and_timeout(self) -> None:
        settings = SimpleNamespace(
            pollinations_api_key="pollinations-test-key",
            pollinations_base_url="https://gen.pollinations.test/v1",
            pollinations_image_timeout_seconds=180,
        )

        with (
            patch.object(deps, "get_settings", return_value=settings),
            patch.object(
                deps,
                "PollinationsClient",
                create=True,
            ) as client_class,
        ):
            deps.get_pollinations_client()

        client_class.assert_called_once_with(
            api_key="pollinations-test-key",
            base_url="https://gen.pollinations.test/v1",
            timeout_seconds=180,
        )

    def test_pollinations_image_service_receives_flux_and_vertical_dimensions(self) -> None:
        settings = SimpleNamespace(
            pollinations_image_model="flux",
            pollinations_image_width=768,
            pollinations_image_height=1024,
        )
        pollinations_client = object()

        with (
            patch.object(deps, "get_settings", return_value=settings),
            patch.object(
                deps,
                "get_pollinations_client",
                return_value=pollinations_client,
                create=True,
            ),
        ):
            service = deps.get_pollinations_image_generation_service()

        self.assertIs(service.pollinations_client, pollinations_client)
        self.assertEqual(service.model, "flux")
        self.assertEqual(service.width, 768)
        self.assertEqual(service.height, 1024)

    def test_image_service_uses_pollinations_as_its_only_image_provider(self) -> None:
        settings = SimpleNamespace(
            output_dir="outputs",
            allow_placeholder_generation=False,
        )
        pollinations_service = SimpleNamespace(model="flux")
        ffmpeg_client = object()

        with (
            patch.object(deps, "get_settings", return_value=settings),
            patch.object(deps, "get_ffmpeg_client", return_value=ffmpeg_client),
            patch.object(
                deps,
                "get_pollinations_image_generation_service",
                return_value=pollinations_service,
                create=True,
            ),
        ):
            service = deps.get_image_service()

        self.assertIs(service.image_generation_service, pollinations_service)
        self.assertEqual(service.image_generation_service.model, "flux")

    def test_tts_service_receives_gemini_primary_and_pollinations_elevenlabs_fallback(self) -> None:
        settings = SimpleNamespace(
            output_dir="outputs",
            gemini_tts_model="gemini-tts-model",
            pollinations_tts_model="elevenlabs",
            allow_placeholder_generation=False,
        )
        gemini_client = object()
        pollinations_client = object()
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
            patch.object(
                deps,
                "get_pollinations_client",
                return_value=pollinations_client,
            ),
        ):
            service = deps.get_tts_service()

        self.assertIs(service.gemini_client, gemini_client)
        self.assertIs(service.pollinations_client, pollinations_client)
        self.assertIs(service.ffmpeg_client, ffmpeg_client)
        self.assertEqual(service.model, "gemini-tts-model")
        self.assertEqual(service.fallback_model, "elevenlabs")


if __name__ == "__main__":
    unittest.main()
