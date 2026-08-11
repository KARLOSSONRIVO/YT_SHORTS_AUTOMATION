import unittest

from app.core.exceptions import IntegrationError
from app.schemas.faceless_video import ScriptGenerationRequest
from app.services.llm_service import LLMService


class FailingTextClient:
    def generate_text(self, **kwargs) -> str:
        raise RuntimeError("provider offline")


class LLMServiceErrorTests(unittest.TestCase):
    def test_unexpected_client_failure_uses_provider_neutral_message(self) -> None:
        service = LLMService(
            llm_client=FailingTextClient(),
            model="llama-3.3-70b-versatile",
            allow_placeholder_generation=False,
        )
        payload = ScriptGenerationRequest(
            job_id="job-1",
            project_id="project-1",
            topic="A forgotten historical event",
        )

        with self.assertRaisesRegex(
            IntegrationError,
            "Story script generation failed: provider offline",
        ):
            service.generate_story_script(payload)


if __name__ == "__main__":
    unittest.main()
