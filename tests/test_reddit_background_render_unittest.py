from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from app.core.exceptions import ValidationError
from app.services.story_render_service import StoryRenderService


class FakeFFmpeg:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def run(self, command: list[str]) -> None:
        self.commands.append(command)


class RedditBackgroundRenderTests(TestCase):
    def test_selects_a_video_from_the_reddit_asset_directory(self) -> None:
        asset_dir = Path(__file__).resolve().parents[1] / "app" / "assets" / "video"
        second = asset_dir / "second.webm"
        service = StoryRenderService(
            ffmpeg_client=None,
            output_dir=asset_dir,
            reddit_story_background_video_dir=asset_dir,
        )

        with patch.object(Path, "iterdir", return_value=[second]), patch.object(Path, "is_file", return_value=True), patch(
            "app.services.story_render_service.random.choice", return_value=second
        ):
            selected = service._select_reddit_background_video()

        self.assertEqual(selected, second.resolve())

    def test_requires_a_video_in_the_reddit_asset_directory(self) -> None:
        asset_dir = Path.cwd() / "missing-reddit-background-assets"
        service = StoryRenderService(
            ffmpeg_client=None,
            output_dir=asset_dir,
            reddit_story_background_video_dir=asset_dir,
        )

        with patch.object(Path, "iterdir", return_value=[]), self.assertRaisesRegex(ValidationError, "Reddit story background video"):
            service._select_reddit_background_video()

    def test_renders_a_random_segment_of_the_background_video(self) -> None:
        base_path = Path(__file__).resolve().parents[1] / "test-output"
        ffmpeg = FakeFFmpeg()
        service = StoryRenderService(ffmpeg_client=ffmpeg, output_dir=base_path)

        with patch.object(service, "_media_duration", return_value=120.0), patch(
            "app.services.story_render_service.random.uniform", return_value=17.5
        ):
            service._render_background_video(
                background_video_path=base_path / "background.mp4",
                audio_path=base_path / "narration.wav",
                subtitles_path=None,
                output_path=base_path / "rendered.mp4",
                duration=60.0,
            )

        command = ffmpeg.commands[0]
        self.assertEqual(command[command.index("-ss") + 1], "17.5")
        self.assertIn(str(base_path / "background.mp4"), command)
        self.assertIn("-stream_loop", command)


if __name__ == "__main__":
    import unittest

    unittest.main()
