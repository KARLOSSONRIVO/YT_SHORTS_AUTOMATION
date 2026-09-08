import json

import pytest

from app.core.exceptions import ValidationError
from app.schemas.faceless_video import ScriptGenerationRequest
from app.services.llm_service import LLMService


def test_reddit_story_prompt_includes_source_and_blocks_historical_substitution():
    captured = {}

    class Client:
        def generate_text(self, *, model, prompt, max_new_tokens, temperature):
            captured["prompt"] = prompt
            return json.dumps({
                "title": "I Slept With My Ex's Best Friend",
                "hook": "The breakup was not the end.",
                "narration": "The breakup was not the end. She made a choice that changed the friendship forever. The confession spread through the group, and everyone had to decide who they believed. The real damage was not the relationship but the trust that disappeared afterward. Some choices cannot be undone, only faced honestly.",
                "caption_text": "A confession changed everything.",
                "scenes": [{
                    "scene_index": 1,
                    "narration": "The breakup was not the end.",
                    "image_prompt": "Vertical 9:16 emotional relationship scene, no text",
                    "duration_seconds": 10,
                    "caption_text": "The breakup was not the end."
                }]
            })

    payload = ScriptGenerationRequest(
        job_id="job-1",
        project_id="project-1",
        topic="Reddit submission from r/confession",
        source_text="I slept with my ex’s best friend a month after we broke up.",
        script_framework="reddit_story",
        target_duration_seconds=15,
    )
    LLMService(llm_client=Client(), model="test").generate_story_script(payload)

    assert "I slept with my ex’s best friend" in captured["prompt"]
    assert "Do not turn this into history" in captured["prompt"]
    assert "Original Reddit submission text is required" in captured["prompt"]


def test_reddit_story_rejects_missing_original_submission_text():
    class Client:
        def generate_text(self, **kwargs):
            raise AssertionError("The model must not be called without source text")

    payload = ScriptGenerationRequest(
        job_id="job-2",
        project_id="project-2",
        topic="Reddit submission from r/confession",
        script_framework="reddit_story",
    )

    with pytest.raises(ValidationError, match="Original Reddit submission text"):
        LLMService(llm_client=Client(), model="test").generate_story_script(payload)
