# Resumable Scene Generation Design

## Problem

Scene generation is all-or-nothing. A Cloudflare failure during a text-cleanup retry aborts the entire request even when a usable image candidate exists, and a restarted request regenerates scene files that were already completed. The provider's useful error body is returned by the Python API but is not emitted in the Python logs, leaving operators with only a generic HTTP 500.

## Design

`ImageService` will treat each scene as an independently resumable unit. A non-empty, decodable existing `scene_XX.png` will be reused; missing or invalid files will be regenerated. This keeps completed work after a later scene fails.

Within a scene, cleanup attempts remain bounded. If an attempt fails after at least one valid candidate was generated, the service will apply the existing final safe-framing cleanup to that candidate and continue. If the first attempt fails, the provider exception will still propagate so genuine configuration and provider failures are visible.

`CloudflareWorkersAIClient` will log failed request metadata and the parsed provider message without logging the API token or full image prompt. HTTP 429 remains a `ProviderRateLimitError`; other HTTP failures remain `IntegrationError`.

## Error Handling

- Never replace a valid completed scene with a failed retry.
- Never silently substitute an image when no candidate was generated.
- Reuse only files Pillow can decode and whose dimensions are non-zero.
- Keep provider status and parsed error message in server logs.
- Do not change the public request or response schema.

## Testing

Focused regression tests will prove that completed scene files are reused, corrupt files are regenerated, a later cleanup failure returns the last valid candidate, and a first-attempt provider failure is still raised. Client tests will verify safe error logging. The full Python test suite will then run before rebuilding and restarting the worker.
