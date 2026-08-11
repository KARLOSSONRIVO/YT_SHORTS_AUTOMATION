from pathlib import Path

from app.services.story_render_service import StoryRenderService


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
