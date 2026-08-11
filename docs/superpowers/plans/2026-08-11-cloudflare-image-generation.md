# Cloudflare Workers AI Scene Image Generation Plan

**Goal:** Replace Gemini scene-image generation with Cloudflare Workers AI model `@cf/bytedance/stable-diffusion-xl-lightning` while retaining Gemini for narration and video generation.

**Architecture:** Add a small REST client for the Workers AI run endpoint and a scene-image generation service that returns the same byte-oriented result expected by `ImageService`. Keep scene processing sequential and surface provider rate limits as `ProviderRateLimitError`; do not silently fall back to Gemini.

**Tech stack:** Python 3.11, httpx, FastAPI dependency wiring, pydantic-settings, unittest.

---

### Task 1: Specify the Cloudflare API boundary

- Add client tests for the model URL, bearer authentication, vertical image payload, binary image response, JSON/base64 response, and HTTP 429 mapping.
- Add routing tests for Cloudflare configuration while preserving the Gemini TTS assertion.
- Run the focused tests and confirm they fail before implementation.

### Task 2: Implement the provider client and service

- Add `CloudflareWorkersAIClient` to call the account-scoped Workers AI REST endpoint.
- Add `CloudflareImageGenerationService` to bind model and generation settings.
- Return generated image bytes and MIME metadata through a provider-specific result object.

### Task 3: Replace the image provider wiring

- Add Cloudflare account, token, model, dimensions, step-count, and base-URL settings.
- Inject the Cloudflare service into `ImageService` using a provider-neutral constructor name.
- Update Docker environment forwarding and README configuration instructions.
- Retain Gemini client wiring for TTS and video only.

### Task 4: Verify and restart

- Run focused tests, the full Python test suite, and static compilation.
- Rebuild and recreate the Python worker container.
- Check container health and report any missing Cloudflare credentials without exposing secrets.
