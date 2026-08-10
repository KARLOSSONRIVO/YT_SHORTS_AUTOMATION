# Script Repair Rate-Limit Recovery Design

## Problem

Story generation may produce a valid but out-of-window script and then make another request to improve its duration. If that later repair request exhausts both the Qwen primary model and the Llama fallback, the current service raises a generic integration error and discards the valid candidate already held in memory. FastAPI returns HTTP 500, so the backend marks the story job unrecoverable.

Job 15 demonstrated this sequence: Qwen returned HTTP 429, Llama returned HTTP 200 with a valid script, then the next repair attempt received HTTP 429 from both models and the job failed.

## Approved Behavior

1. Keep the existing Qwen-to-Llama HTTP 429 fallback.
2. Represent an exhausted Groq HTTP 429 as a typed provider rate-limit error.
3. If a duration-repair request is rate-limited after at least one valid candidate exists, return the closest valid candidate immediately.
4. If rate limiting occurs before any valid candidate exists, return HTTP 429 from the Python API so the backend queue can apply its provider retry policy.
5. Do not recover from authentication failures, malformed responses, missing content, or non-rate-limit provider errors.

## Components

### Provider error

Add a rate-limit-specific application exception with code `provider_rate_limit`. `GroqClient` raises this exception only when its final response remains HTTP 429 after applying the configured fallback model.

### Script service

`LLMService.generate_story_script` catches only the typed rate-limit exception around each provider request. When `best_response` exists, it returns that response. When no candidate exists, it re-raises the exception unchanged.

This preserves the best-effort duration behavior: the service still attempts repairs while quota is available, but a later quota failure cannot erase already-valid work.

### HTTP boundary

The FastAPI application-error handler maps `provider_rate_limit` to HTTP 429. Existing validation errors remain HTTP 422 and other integration errors remain HTTP 500.

The backend already classifies HTTP 429 as a provider rate limit and schedules BullMQ retries. No backend queue-policy change is required.

## Testing

Add regression coverage proving:

- `GroqClient` raises the typed rate-limit exception when both primary and fallback responses are HTTP 429;
- `LLMService` returns its closest valid candidate when the next repair call raises the typed rate-limit exception;
- `LLMService` re-raises the typed rate-limit exception when no valid candidate has been produced; and
- the FastAPI error handler maps the exception to HTTP 429.

Retain all existing provider-routing, duration-repair, invalid-response, and TTS tests. Run the full Python test suite, compile the Python sources, rebuild only the Python worker, and verify both live paths: successful script generation when a valid candidate exists and HTTP 429 when no model can produce one.

## Scope

This change does not alter model names, duration math, prompt content, retry counts, TTS, image generation, rendering, or backend BullMQ retry delays.
