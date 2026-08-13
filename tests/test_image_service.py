from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from app.core.exceptions import ContentSafetyError, IntegrationError
from app.schemas.faceless_video import FacelessScene, SceneImageGenerationRequest
from app.services.image_service import ImageService


def png_bytes(*, color: str = "navy") -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (32, 48), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


class ImageServiceTests(unittest.TestCase):
    def make_service(self, output_dir: str) -> ImageService:
        return ImageService(
            ffmpeg_client=None,
            output_dir=output_dir,
            image_generation_service=SimpleNamespace(
                model="@cf/black-forest-labs/flux-2-dev"
            ),
        )

    def make_payload(self) -> SceneImageGenerationRequest:
        return SceneImageGenerationRequest(
            job_id="job-1",
            project_id="project-1",
            project_title="Resumable Story",
            output_bucket="faceless_story",
            visual_style="cinematic",
            scenes=[
                FacelessScene(
                    scene_index=1,
                    narration="Narration",
                    image_prompt="A quiet office",
                    duration_seconds=6,
                    caption_text="Caption",
                )
            ],
        )

    def test_cleanup_retry_failure_uses_last_valid_candidate(self) -> None:
        service = self.make_service("unused")
        provider_failure = IntegrationError("Cloudflare returned status 400")

        with (
            patch.object(
                service,
                "_generate_image",
                side_effect=[png_bytes(), provider_failure],
            ),
            patch.object(
                service,
                "_processed_image_still_has_text_like_artifact",
                return_value=True,
            ),
            patch.object(
                service,
                "_force_safe_text_free_framing",
                return_value=b"safe-framed-image",
            ),
        ):
            result = service._generate_clean_scene_image("scene prompt")

        self.assertEqual(result, b"safe-framed-image")

    def test_first_generation_failure_is_not_hidden(self) -> None:
        service = self.make_service("unused")

        with patch.object(
            service,
            "_generate_image",
            side_effect=IntegrationError("Cloudflare authentication failed"),
        ):
            with self.assertRaises(IntegrationError) as caught:
                service._generate_clean_scene_image("scene prompt")

        self.assertIn("authentication failed", str(caught.exception))

    def test_tennis_prompt_adds_court_equipment_and_pose_constraints(self) -> None:
        service = self.make_service("unused")

        prompt = service._compose_generation_prompt(
            visual_style="cinematic sports documentary",
            scene_prompt=(
                "An exhausted tennis player collapses after a long rally while "
                "an official approaches"
            ),
        )

        self.assertIn("grass tennis court", prompt)
        self.assertIn("tennis racket", prompt)
        self.assertIn("tennis net", prompt)
        self.assertIn("exactly two legs", prompt)
        self.assertIn("plausible tennis posture", prompt)

    def test_content_safety_rejection_retries_with_neutral_background_prompt(self) -> None:
        service = self.make_service("unused")
        prompts: list[str] = []

        def generate(prompt: str) -> bytes:
            prompts.append(prompt)
            if len(prompts) == 1:
                raise ContentSafetyError("Cloudflare output was flagged")
            return png_bytes(color="teal")

        with (
            patch.object(service, "_generate_image", side_effect=generate),
            patch.object(
                service,
                "_processed_image_still_has_text_like_artifact",
                return_value=False,
            ),
        ):
            result = service._generate_clean_scene_image(
                "Close-up of a well-known baseball star at a scoreboard"
            )

        self.assertTrue(result)
        self.assertEqual(len(prompts), 2)
        self.assertIn("well-known baseball star", prompts[0])
        self.assertIn("adult baseball player", prompts[1])
        self.assertIn("baseball stadium", prompts[1])
        self.assertNotIn("well-known baseball star", prompts[1])
        self.assertNotIn("scoreboard", prompts[1])

    def test_valid_existing_scene_is_reused_without_generation(self) -> None:
        output_dir = Path.cwd()
        service = self.make_service(str(output_dir))
        payload = self.make_payload()
        scene_dir = output_dir / "tests"
        scene_path = scene_dir / "scene_01.png"
        self.addCleanup(scene_path.unlink, missing_ok=True)
        self.addCleanup(scene_path.with_suffix(".generation.json").unlink, missing_ok=True)
        with (
            patch(
                "app.services.image_service.stage_output_dir",
                return_value=scene_dir,
            ),
            patch.object(
                service,
                "_generate_clean_scene_image",
                return_value=png_bytes(color="green"),
            ),
        ):
            first_response = service.generate_scene_images(payload)

        original = scene_path.read_bytes()

        with (
            patch(
                "app.services.image_service.stage_output_dir",
                return_value=scene_dir,
            ),
            patch.object(
                service,
                "_generate_clean_scene_image",
                side_effect=AssertionError("completed scene was regenerated"),
            ),
        ):
            response = service.generate_scene_images(payload)

        self.assertEqual(scene_path.read_bytes(), original)
        self.assertEqual(
            first_response.images[0].image_path,
            response.images[0].image_path,
        )
        self.assertEqual(response.images[0].image_path, str(scene_path.resolve()))

    def test_existing_scene_without_matching_generation_manifest_is_regenerated(self) -> None:
        output_dir = Path.cwd()
        service = self.make_service(str(output_dir))
        payload = self.make_payload()
        scene_dir = output_dir / "tests"
        scene_path = scene_dir / "scene_01.png"
        self.addCleanup(scene_path.unlink, missing_ok=True)
        self.addCleanup(scene_path.with_suffix(".generation.json").unlink, missing_ok=True)
        scene_path.write_bytes(png_bytes(color="green"))
        replacement = png_bytes(color="orange")

        with (
            patch(
                "app.services.image_service.stage_output_dir",
                return_value=scene_dir,
            ),
            patch.object(
                service,
                "_generate_clean_scene_image",
                return_value=replacement,
            ) as generate,
        ):
            service.generate_scene_images(payload)

        self.assertEqual(scene_path.read_bytes(), replacement)
        generate.assert_called_once()

    def test_corrupt_existing_scene_is_regenerated(self) -> None:
        output_dir = Path.cwd()
        service = self.make_service(str(output_dir))
        payload = self.make_payload()
        scene_dir = output_dir / "tests"
        scene_path = scene_dir / "scene_01.png"
        self.addCleanup(scene_path.unlink, missing_ok=True)
        self.addCleanup(scene_path.with_suffix(".generation.json").unlink, missing_ok=True)
        scene_path.write_bytes(b"not-an-image")
        replacement = png_bytes(color="purple")

        with (
            patch(
                "app.services.image_service.stage_output_dir",
                return_value=scene_dir,
            ),
            patch.object(
                service,
                "_generate_clean_scene_image",
                return_value=replacement,
            ) as generate,
        ):
            service.generate_scene_images(payload)

        self.assertEqual(scene_path.read_bytes(), replacement)
        generate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
