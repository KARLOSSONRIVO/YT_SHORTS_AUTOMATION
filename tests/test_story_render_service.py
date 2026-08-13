from pathlib import Path
from unittest.mock import patch

import pytest

from app.core.exceptions import ValidationError
from app.services.story_render_service import StoryRenderService


class FakeFFmpeg:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def run(self, command: list[str]) -> None:
        self.commands.append(command)


def test_concat_file_entry_preserves_normal_path() -> None:
    base_path = Path.cwd() / "test-output"
    path = base_path / "Normal Title" / "scene_01.mp4"

    entry = StoryRenderService._concat_file_entry(path)

    assert entry == f"file '{base_path.resolve().as_posix()}/Normal Title/scene_01.mp4'"


def test_concat_file_entry_escapes_apostrophe_without_truncating_path() -> None:
    base_path = Path.cwd() / "test-output"
    path = base_path / "The World's First Library Burned" / "scene_01.mp4"

    entry = StoryRenderService._concat_file_entry(path)

    assert entry == (
        f"file '{base_path.resolve().as_posix()}/"
        "The World'\\''s First Library Burned/scene_01.mp4'"
    )


def test_subtitle_filter_path_escapes_apostrophe_through_both_parser_layers() -> None:
    base_path = Path.cwd() / "test-output"
    path = base_path / "The World's First Library Burned" / "subtitles.ass"
    service = StoryRenderService(ffmpeg_client=None, output_dir=".")

    escaped_path = service._escape_filter_path(path)

    expected_base = base_path.resolve().as_posix().replace(":", r"\:")
    assert escaped_path == (
        f"{expected_base}/The World'\\\\\\''s First Library Burned/subtitles.ass"
    )


def test_reddit_background_video_selection_uses_the_asset_directory() -> None:
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

    assert selected == second.resolve()


def test_reddit_background_video_requires_an_asset() -> None:
    asset_dir = Path.cwd() / "missing-reddit-background-assets"
    service = StoryRenderService(
        ffmpeg_client=None,
        output_dir=asset_dir,
        reddit_story_background_video_dir=asset_dir,
    )

    with patch.object(Path, "iterdir", return_value=[]), pytest.raises(ValidationError, match="Reddit story background video"):
        service._select_reddit_background_video()


def test_reddit_background_render_uses_a_random_segment_start() -> None:
    base_path = Path(__file__).resolve().parents[1] / "test-output"
    ffmpeg = FakeFFmpeg()
    service = StoryRenderService(ffmpeg_client=ffmpeg, output_dir=base_path)
    background = base_path / "background.mp4"
    audio = base_path / "narration.wav"
    output = base_path / "rendered.mp4"

    with patch.object(service, "_media_duration", return_value=120.0), patch(
        "app.services.story_render_service.random.uniform", return_value=17.5
    ):
        service._render_background_video(
            background_video_path=background,
            audio_path=audio,
            subtitles_path=None,
            output_path=output,
            duration=60.0,
        )

    command = ffmpeg.commands[0]
    assert command[command.index("-ss") + 1] == "17.5"
    assert str(background) in command
    assert "-stream_loop" in command
