from io import BytesIO
from pathlib import Path
import unittest
from unittest.mock import patch

from PIL import Image

from app.core.exceptions import IntegrationError
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
            image_generation_service=None,
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

    def test_valid_existing_scene_is_reused_without_generation(self) -> None:
        output_dir = Path.cwd()
        service = self.make_service(str(output_dir))
        payload = self.make_payload()
        scene_dir = output_dir / "tests"
        scene_path = scene_dir / "scene_01.png"
        self.addCleanup(scene_path.unlink, missing_ok=True)
        original = png_bytes(color="green")
        scene_path.write_bytes(original)

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
        self.assertEqual(response.images[0].image_path, str(scene_path.resolve()))

    def test_corrupt_existing_scene_is_regenerated(self) -> None:
        output_dir = Path.cwd()
        service = self.make_service(str(output_dir))
        payload = self.make_payload()
        scene_dir = output_dir / "tests"
        scene_path = scene_dir / "scene_01.png"
        self.addCleanup(scene_path.unlink, missing_ok=True)
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
