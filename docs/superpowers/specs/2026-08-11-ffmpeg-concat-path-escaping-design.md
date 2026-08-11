# FFmpeg Concat Path Escaping Design

## Goal

Allow faceless-story rendering when a project title produces output paths containing apostrophes, such as `The World's First Library Burned`.

## Root Cause

`StoryRenderService` writes absolute scene paths to FFmpeg's concat-demuxer input file inside single quotes. An apostrophe in a directory name closes that quoted value early, so FFmpeg tries to open a truncated path and returns an integration error.

## Design

Add one focused helper to encode filesystem paths for FFmpeg concat-file syntax. It will normalize the path to POSIX separators and encode an apostrophe by closing the quoted section, escaping the literal apostrophe, and reopening the quoted section. `render_story_video` will use this helper for every `file` entry.

Project titles, directories, URLs, scene files, and the final FFmpeg command remain unchanged. Subtitle-filter escaping is a separate FFmpeg syntax and is outside this change.

## Error Handling

Existing `IntegrationError` behavior remains intact. The change prevents valid apostrophe-bearing paths from being misparsed; genuinely missing or corrupt inputs still fail normally.

## Tests

Add a focused renderer test proving:

1. A normal path remains a valid single-quoted concat entry.
2. `The World's First Library Burned/scene_01.mp4` is encoded as a single logical FFmpeg path rather than truncated at the apostrophe.

Run the focused test first in the failing state, implement the helper, then run the full Python test suite. Finally, rebuild the Python-worker container and rerun the failed render workflow.
