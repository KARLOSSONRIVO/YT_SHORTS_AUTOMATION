# Gemini Image Rate-Limit Recovery Design

## Problem

Gemini scene-image generation returns HTTP 429 when its request or quota limit is exhausted. `GeminiClient` currently converts every non-successful Gemini response into a generic `IntegrationError`, so FastAPI responds with HTTP 500. The backend therefore treats the story-stage failure as permanent instead of applying its existing provider-rate-limit retry policy.

## Design

When the Gemini Interactions API returns HTTP 429, `GeminiClient` will raise `ProviderRateLimitError` with the existing provider error message. The existing FastAPI exception handler will translate that typed error into HTTP 429. The backend already recognizes worker HTTP 429 responses, keeps the project and job queued, and retries with its configured 30, 60, and 120 second backoff.

All other Gemini failures remain `IntegrationError` responses. This preserves permanent handling for malformed requests, invalid credentials, permission failures, safety failures, and invalid provider output.

## Testing

- A Gemini HTTP 429 response raises `ProviderRateLimitError` and retains the provider message.
- A non-429 Gemini error continues to raise `IntegrationError` and is not classified as a provider rate limit.
- The existing FastAPI error-handler test continues to prove that `ProviderRateLimitError` becomes HTTP 429.
- The complete Python test suite must pass before rebuilding the worker.

## Deployment and Verification

Rebuild and recreate only `yt-automation-python-worker`. A controlled request that receives a Gemini rate-limit response must appear to the backend as HTTP 429 rather than HTTP 500, allowing the existing BullMQ retry policy to keep the scene stage queued.
