# FLUX.2 Klein 4B Image Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `@cf/black-forest-labs/flux-2-klein-4b` the default Cloudflare scene-image model and send its documented multipart request without breaking legacy JSON-based image models.

**Architecture:** Keep provider routing and image persistence unchanged. Add model-aware request encoding inside `CloudflareWorkersAIClient`: FLUX.2 Klein uses multipart fields and fixed four-step inference, while existing image models retain the current JSON payload. The existing response decoder remains the shared boundary for binary and base64 responses.

**Tech Stack:** Python 3, httpx, Pydantic Settings, unittest, Docker Compose, Cloudflare Workers AI REST API.

## Global Constraints

- Default model: `@cf/black-forest-labs/flux-2-klein-4b`.
- Scene dimensions remain 1024 by 1792 pixels.
- FLUX.2 Klein requests use `multipart/form-data`.
- Do not send `num_steps` or `negative_prompt` to FLUX.2 Klein.
- Preserve legacy JSON request behavior for other Cloudflare image models.
- Preserve existing network, rate-limit, empty-output, and base64 error handling.
- Do not commit `.env` or any API credential.

---

### Task 1: FLUX.2 multipart request contract

**Files:**
- Modify: `tests/test_cloudflare_workers_ai_client.py`
- Modify: `app/integrations/cloudflare_workers_ai_client.py`

**Interfaces:**
- Consumes: `CloudflareWorkersAIClient.generate_image(model, prompt, negative_prompt, width, height, num_steps, guidance)`.
- Produces: the same `dict[str, Any]` result containing `image_bytes`, `mime_type`, and `provider`.

- [ ] **Step 1: Write the failing multipart regression test**

Replace the SDXL-specific request test with a test that calls:

```python
result = self.make_client(handler).generate_image(
    model="@cf/black-forest-labs/flux-2-klein-4b",
    prompt="cinematic scene",
    negative_prompt="text, watermark",
    width=1024,
    height=1792,
    num_steps=8,
    guidance=7.5,
)
```

Inside the transport handler, assert the endpoint ends in the Klein model ID,
the content type starts with `multipart/form-data; boundary=`, and the encoded
body contains `prompt`, `cinematic scene`, `Avoid: text, watermark`, `width`,
`1024`, `height`, `1792`, `guidance`, and `7.5`. Assert it does not contain the
multipart field names `num_steps` or `negative_prompt`. Return Cloudflare's JSON
envelope with a base64 `result.image`, then assert decoded bytes and MIME type.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
python -m unittest tests.test_cloudflare_workers_ai_client.CloudflareWorkersAIClientTests.test_generate_image_posts_vertical_flux_klein_multipart_payload -v
```

Expected: FAIL because the current client sends `application/json` with SDXL-only fields.

- [ ] **Step 3: Implement model-aware request encoding**

In `CloudflareWorkersAIClient`, identify FLUX.2 Klein models by the exact prefix
`@cf/black-forest-labs/flux-2-klein-`. For those models, append avoidance text
to a non-empty prompt and send:

```python
files = {
    "prompt": (None, flux_prompt),
    "width": (None, str(width)),
    "height": (None, str(height)),
    "guidance": (None, str(guidance)),
}
response = client.post(endpoint, headers=auth_headers, files=files)
```

Do not set `Content-Type`; httpx must add the multipart boundary. Keep the
existing JSON payload and `Content-Type: application/json` for all other models.

- [ ] **Step 4: Run the focused client suite and verify GREEN**

Run:

```powershell
python -m unittest tests.test_cloudflare_workers_ai_client -v
```

Expected: all Cloudflare client tests pass.

### Task 2: Make Klein the configured default

**Files:**
- Modify: `app/core/config.py`
- Modify: `.env`
- Modify: `README.md`
- Modify: `tests/test_provider_routing.py`

**Interfaces:**
- Consumes: `Settings.cloudflare_image_model` and `deps.get_cloudflare_image_generation_service()`.
- Produces: a `CloudflareImageGenerationService` configured with the Klein model and existing 1024-by-1792 scene dimensions.

- [ ] **Step 1: Write failing configuration expectations**

Change the expected model in the settings and dependency-routing tests to the
literal `@cf/black-forest-labs/flux-2-klein-4b` while leaving the dimension,
guidance, and service wiring assertions intact.

- [ ] **Step 2: Run the routing tests and verify RED**

Run:

```powershell
python -m unittest tests.test_provider_routing -v
```

Expected: FAIL because the production default remains SDXL Lightning.

- [ ] **Step 3: Change the runtime and documented defaults**

Set `Settings.cloudflare_image_model`, the local `.env` value, and the README
example to `@cf/black-forest-labs/flux-2-klein-4b`. Update the README prose to
name FLUX.2 Klein 4B and note its multipart/fixed-four-step behavior.

- [ ] **Step 4: Run routing and client tests**

Run:

```powershell
python -m unittest tests.test_provider_routing tests.test_cloudflare_workers_ai_client -v
```

Expected: all selected tests pass.

### Task 3: Full verification and deployment

**Files:**
- Verify: `app/integrations/cloudflare_workers_ai_client.py`
- Verify: `app/core/config.py`
- Verify: `README.md`
- Verify: `tests/test_cloudflare_workers_ai_client.py`
- Verify: `tests/test_provider_routing.py`

**Interfaces:**
- Consumes: the configured Python worker service and Cloudflare credentials from `.env`.
- Produces: a running worker that returns non-empty image bytes from FLUX.2 Klein 4B.

- [ ] **Step 1: Run the complete Python suite**

Run:

```powershell
python -m unittest discover -s tests -v
```

Expected: zero failures and zero errors.

- [ ] **Step 2: Validate the diff**

Run:

```powershell
git diff --check
```

Expected: no whitespace errors.

- [ ] **Step 3: Rebuild and restart the Python worker**

Use the repository's existing Docker Compose service definition to rebuild only
the Python worker, leaving Redis, Mongo, backend services, and existing output
files untouched.

- [ ] **Step 4: Run one live Cloudflare smoke request**

Invoke the configured `CloudflareImageGenerationService` inside the rebuilt
worker with a short cinematic historical-scene prompt. Assert the result model
is `@cf/black-forest-labs/flux-2-klein-4b`, MIME type begins with `image/`, and
the returned byte length is greater than zero. Do not print credentials or the
request authorization header.

- [ ] **Step 5: Verify service health and logs**

Confirm the Python worker is running and its recent logs contain no startup or
Cloudflare request error. Report the exact smoke-test model, MIME type, and image
byte count.

- [ ] **Step 6: Rotate the exposed Cloudflare token**

In the Cloudflare dashboard, revoke the token that appeared in terminal output,
create a replacement with Workers AI Read permission, and replace only the local
`CLOUDFLARE_API_TOKEN` value in `.env`. This credential rotation requires the
user's Cloudflare dashboard access and is not automated by this plan.
