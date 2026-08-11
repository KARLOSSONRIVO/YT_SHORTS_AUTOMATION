# FFmpeg Concat Path Escaping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make faceless-story rendering accept project output paths containing apostrophes and recover failed BullMQ render job `19`.

**Architecture:** Add a small formatter owned by `StoryRenderService` for FFmpeg concat-demuxer file entries. Use it when writing `scene_inputs.txt`, and strengthen the existing subtitle-filter encoder for FFmpeg's two parser layers; keep all output paths unchanged.

**Tech Stack:** Python 3.11+, pytest, FFmpeg, FastAPI, Docker Compose, BullMQ.

## Global Constraints

- Preserve project titles and output directory names exactly.
- Encode paths according to FFmpeg concat-file quoting rules.
- Preserve apostrophes through both concat-demuxer and subtitle-filter parsing.
- Preserve genuinely failing FFmpeg inputs as errors.

---

### Task 1: Escape Concat-Demuxer Paths

**Files:**
- Create: `tests/test_story_render_service.py`
- Modify: `app/services/story_render_service.py:160`
- Modify: `app/services/story_render_service.py:531`

**Interfaces:**
- Consumes: a `pathlib.Path` for a rendered scene clip.
- Produces: `StoryRenderService._concat_file_entry(path: Path) -> str`, a complete FFmpeg concat `file` directive.
- Produces: `_escape_filter_path(path: Path) -> str` output that preserves apostrophes through both FFmpeg filter parser layers.

- [x] **Step 1: Write the failing tests**

```python
from pathlib import Path

from app.services.story_render_service import StoryRenderService


def test_concat_file_entry_preserves_normal_path() -> None:
    entry = StoryRenderService._concat_file_entry(Path("/tmp/Normal Title/scene_01.mp4"))
    assert entry == "file '/tmp/Normal Title/scene_01.mp4'"


def test_concat_file_entry_escapes_apostrophe_without_truncating_path() -> None:
    entry = StoryRenderService._concat_file_entry(
        Path("/tmp/The World's First Library Burned/scene_01.mp4")
    )
    assert entry == "file '/tmp/The World'\\''s First Library Burned/scene_01.mp4'"
```

- [x] **Step 2: Run the focused tests and verify RED**

Run: `pytest -q tests/test_story_render_service.py`

Expected: FAIL because `StoryRenderService._concat_file_entry` does not exist.

- [x] **Step 3: Implement the minimal formatter and use it**

```python
    @staticmethod
    def _concat_file_entry(path: Path) -> str:
        normalized_path = path.resolve().as_posix()
        escaped_path = normalized_path.replace("'", r"'\''")
        return f"file '{escaped_path}'"
```

Replace the current concat-line construction with:

```python
concat_lines.append(self._concat_file_entry(scene_clip_path))
```

- [x] **Step 4: Run the focused tests and verify GREEN**

Run: `pytest -q tests/test_story_render_service.py`

Expected: 2 passed.

- [x] **Step 5: Run full verification**

Run: `pytest -q`

Expected: all Python tests pass.

Run: `python -m compileall -q app tests`

Expected: exit code 0.

Run: `git diff --check`

Expected: no whitespace errors.

- [x] **Step 6: Rebuild and recover the failed render**

Run: `docker compose up -d --build python-worker`

Expected: `yt-automation-python-worker` is recreated and healthy.

Retry BullMQ story job `19` through BullMQ's `Job.retry()` API from the backend-worker container. Then follow the Python-worker and backend-worker logs until the render completes or a new concrete error appears.

- [x] **Step 7: Preserve implementation safely**

The renderer source already contains broader uncommitted user changes. Do not commit the implementation if staging it would capture unrelated work; report the workspace state instead.
