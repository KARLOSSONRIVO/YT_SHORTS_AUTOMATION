# Best-Effort Script Duration Design

## Problem

The Python story generator currently treats the requested duration as a hard acceptance gate. For a 60-second project, each parsed script must fall between 90% and 108% of the target. When all three valid model responses fall outside that narrow window, the service raises an integration error and the story queue records an HTTP 500.

This behavior discards usable scripts and blocks the rest of the pipeline even when Qwen or the Llama fallback successfully returned valid structured content.

## Approved Behavior

Duration is a best-effort optimization, not a fatal validation rule.

1. Generate up to three valid script candidates using the existing primary and provider-fallback behavior.
2. Return immediately when a candidate is inside the preferred duration window.
3. Otherwise retain the candidate whose estimated narration duration is closest to the requested target.
4. After three valid but out-of-window candidates, return the closest candidate instead of raising an HTTP 500.
5. Scripts shorter or longer than 60 seconds are allowed when they are the closest valid result.

## Error Handling

The change does not hide real generation failures. The request must still fail when:

- Groq and its configured fallback return an HTTP error;
- the provider response is not usable JSON;
- the parsed response contains no usable scenes or narration; or
- another integration error prevents construction of a valid script candidate.

Rate-limit fallback remains `qwen/qwen3.6-27b` to `llama-3.1-8b-instant`. This design changes only what happens after valid candidates miss the preferred duration window.

## Implementation

`LLMService.generate_story_script` will track both the latest candidate used to build the next repair prompt and the best candidate seen so far. "Best" means the smallest absolute difference between estimated narration duration and `target_duration_seconds`.

The existing retry prompts and strict preferred window remain unchanged so the service continues trying to produce a well-timed script. Only the terminal behavior changes: return the best candidate rather than raising `IntegrationError` solely because duration remained outside the window.

## Tests

Add a regression test with three valid out-of-window responses. It must prove that:

- all three repair attempts occur;
- the candidate closest to the target is returned, even when it is longer than 60 seconds; and
- no duration-only integration error is raised.

Retain the existing tests for in-window repair, malformed responses, provider failures, model routing, and rate-limit fallback. Run the full Python unit test suite, compile the Python sources, rebuild the worker image, and verify a real `/internal/faceless/generate-script` request no longer returns a duration-only HTTP 500.

## Scope

This change does not alter target-duration settings, speaking-rate math, Groq model selection, TTS, image generation, rendering, or queue retry policy.
