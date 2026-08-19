import json
import unittest

from app.core.exceptions import IntegrationError, ProviderRateLimitError
from app.schemas.faceless_video import ScriptGenerationRequest
from app.services.llm_service import LLMService


HOOK = "History almost forgot this."
SHORT_NARRATION = f"{HOOK} {' '.join(['detail'] * 50)}"
VALID_NARRATION = f"{HOOK} {' '.join(['detail'] * 116)}"
LONG_NARRATION = f"{HOOK} {' '.join(['detail'] * 131)}"
TARGET_NARRATION = f"{HOOK} {' '.join(['detail'] * 120)}"
OVERSIZED_TOP_LEVEL_NARRATION = f"{HOOK} {' '.join(['detail'] * 181)}"
FAR_SHORT_NARRATION = f"{HOOK} {' '.join(['detail'] * 50)}"
FAR_LONG_NARRATION = f"{HOOK} {' '.join(['detail'] * 150)}"
CLOSEST_LONG_NARRATION = f"{HOOK} {' '.join(['detail'] * 140)}"


def script_response(narration: str) -> str:
    words = narration.split()
    base_scene_words, extra_words = divmod(len(words), 6)
    scenes = []
    offset = 0
    for index in range(1, 7):
        scene_word_count = base_scene_words + (1 if index <= extra_words else 0)
        scene_narration = " ".join(words[offset : offset + scene_word_count])
        offset += scene_word_count
        scenes.append(
            {
                "scene_index": index,
                "narration": scene_narration,
                "image_prompt": f"Cinematic historical scene {index}",
                "duration_seconds": 10,
                "caption_text": "Historical detail",
            }
        )
    return json.dumps(
        {
            "title": "A forgotten Philippine story",
            "hook": HOOK,
            "narration": narration,
            "caption_text": "A forgotten story",
            "scenes": scenes,
        }
    )


def mismatched_narration_response() -> str:
    scene_word_counts = [19, 18, 18, 18, 19, 19]
    scenes = []
    for index, word_count in enumerate(scene_word_counts, start=1):
        if index == 1:
            narration = f"{HOOK} {' '.join(['detail'] * (word_count - 4))}"
        else:
            narration = " ".join(["detail"] * word_count)
        scenes.append(
            {
                "scene_index": index,
                "narration": narration,
                "image_prompt": f"Cinematic historical scene {index}",
                "duration_seconds": 10,
                "caption_text": "Historical detail",
            }
        )
    return json.dumps(
        {
            "title": "A forgotten Philippine story",
            "hook": HOOK,
            "narration": OVERSIZED_TOP_LEVEL_NARRATION,
            "caption_text": "A forgotten story",
            "scenes": scenes,
        }
    )


class FeedbackAwareTextClient:
    def generate_text(self, **kwargs) -> str:
        prompt = kwargs["prompt"]
        has_measured_feedback = all(
            marker in prompt
            for marker in (
                "Previous narration word count: 54",
                "Acceptable full narration range: 112 to 133 spoken words",
                "Previous narration to revise:",
                SHORT_NARRATION,
            )
        )
        return script_response(VALID_NARRATION if has_measured_feedback else SHORT_NARRATION)


class OvershootRepairTextClient:
    def generate_text(self, **kwargs) -> str:
        prompt = kwargs["prompt"]
        has_target_headroom = all(
            marker in prompt
            for marker in (
                "Previous narration word count: 135",
                "Target narration word count: 124",
                "Remove at least 11 lower-value spoken words",
                LONG_NARRATION,
            )
        )
        return script_response(TARGET_NARRATION if has_target_headroom else LONG_NARRATION)


class SceneAwareRepairTextClient:
    def generate_text(self, **kwargs) -> str:
        prompt = kwargs["prompt"]
        if "Previous narration word count: 111" in prompt:
            return script_response(TARGET_NARRATION)
        return mismatched_narration_response()


class AlwaysOutOfWindowTextClient:
    def __init__(self) -> None:
        self.calls = 0
        self.max_new_tokens: list[int] = []
        self.responses = (
            FAR_SHORT_NARRATION,
            FAR_LONG_NARRATION,
            CLOSEST_LONG_NARRATION,
        )

    def generate_text(self, **kwargs) -> str:
        self.max_new_tokens.append(kwargs["max_new_tokens"])
        narration = self.responses[self.calls]
        self.calls += 1
        return script_response(narration)


class RateLimitedAfterCandidateTextClient:
    def __init__(self) -> None:
        self.calls = 0

    def generate_text(self, **kwargs) -> str:
        self.calls += 1
        if self.calls == 1:
            return script_response(FAR_SHORT_NARRATION)
        raise ProviderRateLimitError("Groq quota reached")


class AlwaysRateLimitedTextClient:
    def generate_text(self, **kwargs) -> str:
        raise ProviderRateLimitError("Groq quota reached")


class InvalidFallbackAfterCandidateTextClient:
    def __init__(self) -> None:
        self.calls = 0

    def generate_text(self, **kwargs) -> str:
        self.calls += 1
        if self.calls == 1:
            return script_response(FAR_SHORT_NARRATION)
        raise IntegrationError(
            "Groq text API failed with status 400: Failed to validate JSON."
        )


class LLMServiceDurationTests(unittest.TestCase):
    def test_short_history_script_is_repaired_with_measured_duration_feedback(self) -> None:
        service = LLMService(
            llm_client=FeedbackAwareTextClient(),
            model="llama-3.3-70b-versatile",
            allow_placeholder_generation=False,
        )
        payload = ScriptGenerationRequest(
            job_id="job-duration",
            project_id="project-duration",
            topic="The Life and Legacy of Emilio Aguinaldo",
            target_duration_seconds=60,
            speaking_rate=0.96,
            script_framework="history_story",
            story_format="mystery_reveal",
        )

        result = service.generate_story_script(payload)

        self.assertEqual(len(result.narration.split()), 120)

    def test_long_history_script_is_repaired_to_center_target_with_headroom(self) -> None:
        service = LLMService(
            llm_client=OvershootRepairTextClient(),
            model="llama-3.3-70b-versatile",
            allow_placeholder_generation=False,
        )
        payload = ScriptGenerationRequest(
            job_id="job-overshoot",
            project_id="project-overshoot",
            topic="The Life and Legacy of Emilio Aguinaldo",
            target_duration_seconds=60,
            speaking_rate=0.96,
            script_framework="history_story",
            story_format="hero_story",
        )

        result = service.generate_story_script(payload)

        self.assertEqual(len(result.narration.split()), 124)

    def test_scene_narration_drives_repair_when_top_level_narration_is_oversized(self) -> None:
        service = LLMService(
            llm_client=SceneAwareRepairTextClient(),
            model="llama-3.3-70b-versatile",
            allow_placeholder_generation=False,
        )
        payload = ScriptGenerationRequest(
            job_id="job-scene-source",
            project_id="project-scene-source",
            topic="The Life and Legacy of Emilio Aguinaldo",
            target_duration_seconds=60,
            speaking_rate=0.96,
            script_framework="history_story",
            story_format="hero_story",
        )

        result = service.generate_story_script(payload)

        self.assertEqual(len(result.narration.split()), 124)

    def test_closest_valid_script_is_returned_after_duration_retries(self) -> None:
        client = AlwaysOutOfWindowTextClient()
        service = LLMService(
            llm_client=client,
            model="qwen/qwen3.6-27b",
            allow_placeholder_generation=False,
        )
        payload = ScriptGenerationRequest(
            job_id="job-best-effort-duration",
            project_id="project-best-effort-duration",
            topic="The Mystery of the Laguna Copperplate Inscription",
            target_duration_seconds=60,
            speaking_rate=0.96,
            script_framework="history_story",
            story_format="mystery_reveal",
        )

        result = service.generate_story_script(payload)

        self.assertEqual(client.calls, 2)
        self.assertEqual(client.max_new_tokens, [1600, 1600])
        self.assertEqual(result.narration, FAR_LONG_NARRATION)
        self.assertGreater(
            service._estimate_narration_duration_seconds(
                result.narration,
                payload.speaking_rate,
            ),
            payload.target_duration_seconds * service.MAX_DURATION_RATIO,
        )

    def test_existing_candidate_is_returned_when_fallback_rejects_json(self) -> None:
        client = InvalidFallbackAfterCandidateTextClient()
        service = LLMService(
            llm_client=client,
            model="qwen/qwen3.6-27b",
            allow_placeholder_generation=False,
        )
        payload = ScriptGenerationRequest(
            job_id="job-fallback-json",
            project_id="project-fallback-json",
            topic="The Mystery of the Laguna Copperplate Inscription",
            target_duration_seconds=60,
            speaking_rate=0.96,
            script_framework="history_story",
            story_format="mystery_reveal",
        )

        result = service.generate_story_script(payload)

        self.assertEqual(client.calls, 2)
        self.assertEqual(result.narration, FAR_SHORT_NARRATION)

    def test_existing_candidate_is_returned_when_duration_repair_is_rate_limited(self) -> None:
        client = RateLimitedAfterCandidateTextClient()
        service = LLMService(
            llm_client=client,
            model="qwen/qwen3.6-27b",
            allow_placeholder_generation=False,
        )
        payload = ScriptGenerationRequest(
            job_id="job-repair-rate-limit",
            project_id="project-repair-rate-limit",
            topic="The Mystery of the Laguna Copperplate Inscription",
            target_duration_seconds=60,
            speaking_rate=0.96,
            script_framework="history_story",
            story_format="mystery_reveal",
        )

        result = service.generate_story_script(payload)

        self.assertEqual(client.calls, 2)
        self.assertEqual(result.narration, FAR_SHORT_NARRATION)

    def test_rate_limit_before_any_candidate_is_re_raised(self) -> None:
        service = LLMService(
            llm_client=AlwaysRateLimitedTextClient(),
            model="qwen/qwen3.6-27b",
            allow_placeholder_generation=False,
        )
        payload = ScriptGenerationRequest(
            job_id="job-initial-rate-limit",
            project_id="project-initial-rate-limit",
            topic="The Mystery of the Laguna Copperplate Inscription",
            target_duration_seconds=60,
            speaking_rate=0.96,
            script_framework="history_story",
            story_format="mystery_reveal",
        )

        with self.assertRaises(ProviderRateLimitError):
            service.generate_story_script(payload)


if __name__ == "__main__":
    unittest.main()
