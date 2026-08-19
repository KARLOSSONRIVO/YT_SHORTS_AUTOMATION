from __future__ import annotations

from array import array
from io import BytesIO
import json
import math
from pathlib import Path
import tempfile
import unittest
import wave

import httpx

from app.core.exceptions import IntegrationError, ProviderRateLimitError
from app.integrations.pollinations_client import PollinationsClient
from app.schemas.faceless_video import AudioGenerationRequest
from app.services.tts_service import TTSService


def wav_bytes(*, amplitude: int = 0, frames: int = 240) -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24_000)
        wav_file.writeframes(array("h", [amplitude] * frames).tobytes())
    return output.getvalue()


class PollinationsSpeechClientTests(unittest.TestCase):
    def make_client(self, handler) -> PollinationsClient:
        return PollinationsClient(
            api_key="pollinations-test-key",
            base_url="https://gen.pollinations.test/v1",
            timeout_seconds=30,
            transport=httpx.MockTransport(handler),
        )

    def test_generate_speech_posts_elevenlabs_voice_without_qwen_instruction(self) -> None:
        expected_audio = wav_bytes()

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(
                str(request.url),
                "https://gen.pollinations.test/v1/audio/speech",
            )
            self.assertEqual(
                request.headers["Authorization"],
                "Bearer pollinations-test-key",
            )
            payload = json.loads(request.content)
            self.assertEqual(payload["model"], "elevenlabs")
            self.assertEqual(payload["input"], "[sorrowful] The room fell silent.")
            self.assertEqual(payload["voice"], "james")
            self.assertEqual(payload["response_format"], "wav")
            self.assertNotIn("instruct", payload)
            return httpx.Response(
                200,
                content=expected_audio,
                headers={"Content-Type": "audio/wav"},
            )

        result = self.make_client(handler).generate_speech(
            model="elevenlabs",
            text="[sorrowful] The room fell silent.",
            voice="james",
            instruct="",
            speed=0.94,
            seed=1936,
        )

        self.assertEqual(result, expected_audio)

    def test_generate_speech_sends_speed_and_seed_for_consistent_delivery(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            self.assertEqual(payload["speed"], 0.94)
            self.assertEqual(payload["seed"], 1936)
            return httpx.Response(200, content=wav_bytes())

        self.make_client(handler).generate_speech(
            model="elevenlabs",
            text="A careful decision begins with one clear step.",
            voice="brian",
            instruct="",
            speed=0.94,
            seed=1936,
        )

    def test_generate_speech_maps_429_to_provider_rate_limit(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429,
                json={"error": {"message": "audio rate limit exceeded"}},
            )

        with self.assertRaises(ProviderRateLimitError) as caught:
            self.make_client(handler).generate_speech(
                model="elevenlabs",
                text="Narration",
                voice="brian",
                instruct="",
                speed=1.0,
                seed=1936,
            )

        self.assertIn("status 429", str(caught.exception))
        self.assertIn("audio rate limit exceeded", str(caught.exception))


class GeminiRateLimitTTSFallbackTests(unittest.TestCase):
    def request(self) -> AudioGenerationRequest:
        return AudioGenerationRequest(
            job_id="audio-job",
            project_id="project-1",
            project_title="A Story",
            output_bucket="faceless_story",
            narration="  The room fell silent.  ",
            voice="Kore",
            speaking_rate=0.82,
        )

    def service(
        self,
        *,
        output_dir: str,
        gemini_client,
        pollinations_client,
    ) -> TTSService:
        return TTSService(
            gemini_client=gemini_client,
            pollinations_client=pollinations_client,
            ffmpeg_client=object(),
            output_dir=output_dir,
            model="gemini-tts-model",
            fallback_model="elevenlabs",
        )

    def test_gemini_429_uses_elevenlabs_v3_and_reports_actual_voice(self) -> None:
        gemini_error = ProviderRateLimitError("Gemini daily quota exhausted")
        gemini = FakeGeminiClient(error=gemini_error)
        pollinations = FakePollinationsSpeechClient(result=wav_bytes())

        with tempfile.TemporaryDirectory() as output_dir:
            result = self.service(
                output_dir=output_dir,
                gemini_client=gemini,
                pollinations_client=pollinations,
            ).generate_narration(self.request())

            self.assertEqual(Path(result.audio_path).read_bytes(), pollinations.result)

        self.assertEqual(result.voice, "brian")
        self.assertEqual(len(pollinations.calls), 1)
        call = pollinations.calls[0]
        self.assertEqual(call["model"], "elevenlabs")
        self.assertEqual(call["text"], "[calm] The room fell silent.")
        self.assertEqual(call["voice"], "brian")
        self.assertIsNone(call["instruct"])
        self.assertEqual(call["speed"], 0.82)

    def test_non_rate_limit_gemini_error_does_not_use_pollinations(self) -> None:
        gemini_error = IntegrationError("Gemini authentication failed")
        gemini = FakeGeminiClient(error=gemini_error)
        pollinations = FakePollinationsSpeechClient(result=wav_bytes())

        with tempfile.TemporaryDirectory() as output_dir:
            with self.assertRaises(IntegrationError) as caught:
                self.service(
                    output_dir=output_dir,
                    gemini_client=gemini,
                    pollinations_client=pollinations,
                ).generate_narration(self.request())

        self.assertIs(caught.exception, gemini_error)
        self.assertEqual(pollinations.calls, [])

    def test_unconfigured_pollinations_preserves_gemini_429(self) -> None:
        gemini_error = ProviderRateLimitError("Gemini daily quota exhausted")
        gemini = FakeGeminiClient(error=gemini_error)
        pollinations = FakePollinationsSpeechClient(
            configured=False,
            result=wav_bytes(),
        )

        with tempfile.TemporaryDirectory() as output_dir:
            with self.assertRaises(ProviderRateLimitError) as caught:
                self.service(
                    output_dir=output_dir,
                    gemini_client=gemini,
                    pollinations_client=pollinations,
                ).generate_narration(self.request())

        self.assertIs(caught.exception, gemini_error)
        self.assertEqual(pollinations.calls, [])

    def test_elevenlabs_voice_is_selected_from_the_story(self) -> None:
        gemini = FakeGeminiClient(
            error=ProviderRateLimitError("Gemini daily quota exhausted")
        )
        pollinations = FakePollinationsSpeechClient(result=wav_bytes())

        with tempfile.TemporaryDirectory() as output_dir:
            service = self.service(
                output_dir=output_dir,
                gemini_client=gemini,
                pollinations_client=pollinations,
            )
            psychology_request = self.request().model_copy(
                update={
                    "project_id": "psychology-story",
                    "voice": "Achernar",
                    "narration": (
                        "Your brain is not weak. Fear makes every decision feel "
                        "dangerous, but clarity begins when you choose one step."
                    ),
                }
            )
            history_request = self.request().model_copy(
                update={
                    "project_id": "history-story",
                    "narration": (
                        "The warrior queen led her army against an empire. "
                        "History tried to erase her victory."
                    ),
                }
            )

            psychology_result = service.generate_narration(psychology_request)
            history_result = service.generate_narration(history_request)

        self.assertEqual(pollinations.calls[0]["voice"], "brian")
        self.assertEqual(psychology_result.voice, "brian")
        self.assertEqual(pollinations.calls[1]["voice"], "george")
        self.assertEqual(history_result.voice, "george")
        self.assertNotEqual(
            pollinations.calls[0]["voice"],
            pollinations.calls[1]["voice"],
        )

    def test_elevenlabs_v3_adds_emotional_beats_without_qwen_instruct(self) -> None:
        gemini = FakeGeminiClient(
            error=ProviderRateLimitError("Gemini daily quota exhausted")
        )
        pollinations = FakePollinationsSpeechClient(result=wav_bytes())
        narration = (
            "Your brain hates choosing. It treats every decision as a threat. "
            "But here is the twist. The pain comes from fearing the wrong answer. "
            "Instead, choose one good option. Clarity comes after action."
        )

        with tempfile.TemporaryDirectory() as output_dir:
            self.service(
                output_dir=output_dir,
                gemini_client=gemini,
                pollinations_client=pollinations,
            ).generate_narration(self.request().model_copy(update={"narration": narration}))

        call = pollinations.calls[0]
        self.assertIn("[curious]", call["text"])
        self.assertIn("[pause]", call["text"])
        self.assertIn("[determined]", call["text"])
        self.assertIsNone(call["instruct"])
        directed_words = str(call["text"])
        for tag in ("[curious] ", "[pause] ", "[determined] "):
            directed_words = directed_words.replace(tag, "")
        self.assertEqual(directed_words, narration)

    def test_elevenlabs_fallback_chunks_only_beyond_api_limit_and_joins_wav(self) -> None:
        gemini = FakeGeminiClient(
            error=ProviderRateLimitError("Gemini daily quota exhausted")
        )
        pollinations = FakePollinationsSpeechClient(result=wav_bytes())
        long_narration = " ".join(
            f"Sentence {index} reveals another important detail."
            for index in range(160)
        )
        request = self.request().model_copy(update={"narration": long_narration})

        with tempfile.TemporaryDirectory() as output_dir:
            result = self.service(
                output_dir=output_dir,
                gemini_client=gemini,
                pollinations_client=pollinations,
            ).generate_narration(request)

            with wave.open(result.audio_path, "rb") as combined:
                combined_frames = combined.getnframes()

        self.assertGreater(len(pollinations.calls), 1)
        self.assertTrue(
            all(len(str(call["text"])) <= 4_096 for call in pollinations.calls)
        )
        expected_pause_frames = 2_880 * (len(pollinations.calls) - 1)
        self.assertEqual(
            combined_frames,
            240 * len(pollinations.calls) + expected_pause_frames,
        )
        self.assertEqual(
            " ".join(
                str(call["text"])
                .replace("[calm] ", "")
                .replace("[pause] ", "")
                .replace("[determined] ", "")
                for call in pollinations.calls
            ),
            long_narration,
        )
        self.assertTrue(all(call["speed"] == 0.82 for call in pollinations.calls))
        self.assertEqual(
            len({call["seed"] for call in pollinations.calls}),
            1,
        )
        self.assertTrue(all(call["voice"] == "brian" for call in pollinations.calls))
        self.assertTrue(all(call["instruct"] is None for call in pollinations.calls))

    def test_elevenlabs_fallback_raises_quiet_audio_to_reference_level(self) -> None:
        gemini = FakeGeminiClient(
            error=ProviderRateLimitError("Gemini daily quota exhausted")
        )
        pollinations = FakePollinationsSpeechClient(
            result=wav_bytes(amplitude=1_000, frames=24_000)
        )

        with tempfile.TemporaryDirectory() as output_dir:
            result = self.service(
                output_dir=output_dir,
                gemini_client=gemini,
                pollinations_client=pollinations,
            ).generate_narration(self.request())

            with wave.open(result.audio_path, "rb") as normalized:
                samples = array("h", normalized.readframes(normalized.getnframes()))

        rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
        rms_dbfs = 20 * math.log10(rms / 32_767)
        self.assertAlmostEqual(rms_dbfs, -19.0, delta=0.25)
        self.assertLessEqual(max(abs(sample) for sample in samples), 27_567)


class FakeGeminiClient:
    def __init__(self, *, result: bytes | None = None, error: Exception | None = None):
        self.result = result
        self.error = error
        self.calls: list[dict[str, str]] = []

    def generate_speech(self, *, model: str, text: str, voice: str) -> bytes:
        self.calls.append({"model": model, "text": text, "voice": voice})
        if self.error:
            raise self.error
        return self.result or wav_bytes()


class FakePollinationsSpeechClient:
    def __init__(
        self,
        *,
        configured: bool = True,
        result: bytes,
    ) -> None:
        self.configured = configured
        self.result = result
        self.calls: list[dict[str, str]] = []

    def is_configured(self) -> bool:
        return self.configured

    def generate_speech(
        self,
        *,
        model: str,
        text: str,
        voice: str,
        instruct: str | None,
        speed: float,
        seed: int,
    ) -> bytes:
        self.calls.append(
            {
                "model": model,
                "text": text,
                "voice": voice,
                "instruct": instruct,
                "speed": speed,
                "seed": seed,
            }
        )
        return self.result


if __name__ == "__main__":
    unittest.main()
