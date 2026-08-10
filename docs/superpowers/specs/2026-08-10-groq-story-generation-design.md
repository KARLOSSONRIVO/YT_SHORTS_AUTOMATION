# Groq Story Generation Design

Date: 2026-08-10

## Goal

Route story and script generation through Groq while keeping Gemini responsible for image generation and text-to-speech. Groq is required for scripts and does not fall back to Gemini.

## Provider Responsibilities

- Groq generates story scripts through `LLMService`.
- Gemini generates scene images through `GeminiImageGenerationService`.
- Gemini generates narration audio through `TTSService`.
- Existing Gemini video-generation behavior remains unchanged.

This division keeps each provider explicit at dependency-injection boundaries and prevents media generation from being affected by the script-provider change.

## Groq Integration

Add a focused `GroqClient` alongside `GeminiClient`. It will call Groq's OpenAI-compatible chat-completions endpoint with the project's existing `httpx` dependency and expose the `generate_text` interface already consumed by `LLMService`.

The client request will:

- authenticate with `GROQ_API_KEY`;
- use the configured `GROQ_MODEL`;
- send the existing script prompt as a user message;
- map `max_new_tokens` to `max_completion_tokens`;
- pass through the configured temperature;
- request JSON Object Mode; and
- return the first assistant message's text content.

The default model is `llama-3.3-70b-versatile`. It is a Groq production model with strong multilingual natural-language performance and JSON Object Mode, making it a suitable default for creative, structured short-form scripts.

## Configuration

Add these settings:

- `groq_api_key`, accepting `PY_WORKER_GROQ_API_KEY` or `GROQ_API_KEY`;
- `groq_model`, accepting `PY_WORKER_GROQ_MODEL` or `GROQ_MODEL`, defaulting to `llama-3.3-70b-versatile`; and
- `groq_base_url`, accepting `PY_WORKER_GROQ_BASE_URL` or `GROQ_BASE_URL`, defaulting to `https://api.groq.com/openai/v1`.

Existing Gemini settings remain the source of image, speech, and video model configuration. The generic Gemini text-model setting can remain for backward configuration compatibility, but story dependency wiring will no longer consume it.

## Dependency Wiring and Data Flow

1. The script-generation route resolves `LLMService`.
2. `get_llm_service` injects the cached `GroqClient` and `groq_model`.
3. `LLMService` builds the existing retention-focused prompt and calls `GroqClient.generate_text`.
4. `GroqClient` requests a JSON object from Groq.
5. `LLMService` parses, normalizes, duration-checks, and retries the result using its existing behavior.
6. Later workflow stages continue resolving Gemini-backed image and TTS services independently.

No Gemini fallback occurs during story generation.

## Error Handling

`GroqClient` raises `IntegrationError` when:

- `GROQ_API_KEY` is missing;
- the model setting is empty;
- Groq returns an HTTP error;
- Groq returns non-JSON transport data; or
- the response contains no assistant text.

Error messages identify Groq without exposing the API key. `LLMService` changes its hard-coded Gemini failure wording to provider-neutral story-generation wording while preserving placeholder behavior when that existing option is enabled.

## Compatibility and Scope

- Do not change public API request or response schemas.
- Do not change prompts, duration tuning, scene parsing, image generation, TTS behavior, or rendering behavior.
- Do not add Groq or OpenAI SDK dependencies; use the existing HTTP client.
- Preserve all current uncommitted changes in overlapping files.

## Testing

Follow test-driven development:

1. Verify `GroqClient` sends authentication, model, prompt, token limit, temperature, and JSON Object Mode correctly.
2. Verify Groq HTTP errors and malformed or empty responses become clear `IntegrationError` failures.
3. Verify settings load `GROQ_API_KEY` and the default Groq model.
4. Verify dependency wiring gives `LLMService` the Groq client while Gemini image and TTS services still receive the Gemini client.
5. Run the focused tests, full Python test suite, and a Python compile/import check.

No live provider request is required for automated tests; HTTP transport is isolated at the client boundary.
