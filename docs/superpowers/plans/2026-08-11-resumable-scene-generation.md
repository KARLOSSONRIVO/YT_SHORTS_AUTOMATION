# Resumable Scene Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Cloudflare scene generation resume completed work and survive a failed cleanup retry when a usable candidate already exists.

**Architecture:** Keep recovery local to `ImageService`, where scene files and generated candidates are available. Add safe provider-failure logging at the Cloudflare client boundary without changing API schemas or exposing secrets.

**Tech Stack:** Python 3, FastAPI, httpx, Pillow, unittest/pytest

## Global Constraints

- Preserve all unrelated dirty-worktree changes.
- Do not log the Cloudflare API token or complete prompts.
- A first-attempt provider failure must remain fatal.
- Existing valid scene PNGs must not invoke image generation again.

---

### Task 1: Recover a usable candidate from a failed cleanup retry

**Files:**
- Modify: `app/services/image_service.py`
- Create: `tests/test_image_service.py`

**Interfaces:**
- Consumes: `ImageService._generate_image(prompt: str) -> bytes`
- Produces: `ImageService._generate_clean_scene_image(prompt: str) -> bytes`

- [ ] Write a test whose first generated candidate triggers cleanup and whose second provider call raises `IntegrationError`; assert that valid image bytes are returned.
- [ ] Run `python -m pytest tests/test_image_service.py -v` and confirm the recovery test fails by raising the provider exception.
- [ ] Catch per-attempt failures only when `last_candidate` exists and return the existing final safe-framed candidate.
- [ ] Add a test proving a first-attempt failure still raises, then run the focused tests.

### Task 2: Resume completed scene files

**Files:**
- Modify: `app/services/image_service.py`
- Modify: `tests/test_image_service.py`

**Interfaces:**
- Consumes: a target `Path` for `scene_XX.png`
- Produces: a boolean validity decision used by `generate_scene_images`

- [ ] Write a test with one valid pre-existing PNG and assert the generator is not invoked for that scene.
- [ ] Run the test and confirm it fails because the file is overwritten.
- [ ] Add a focused helper that validates an existing image with Pillow and reuse it in `generate_scene_images`.
- [ ] Write a corrupt-file test that asserts regeneration occurs, then run the focused tests.

### Task 3: Emit safe Cloudflare provider errors

**Files:**
- Modify: `app/integrations/cloudflare_workers_ai_client.py`
- Modify: `tests/test_cloudflare_workers_ai_client.py`

**Interfaces:**
- Consumes: `httpx.Response` and parsed provider error message
- Produces: one error log containing model, HTTP status, and provider message

- [ ] Write a log-capture test for a Cloudflare HTTP 400 response and assert status/message/model are logged while token and prompt are absent.
- [ ] Run the client test and confirm it fails due to the missing error log.
- [ ] Add the minimal logger call before exception classification.
- [ ] Run the focused client tests.

### Task 4: Verify and deploy locally

**Files:**
- No additional source files.

**Interfaces:**
- Consumes: updated Python worker source and tests
- Produces: rebuilt and restarted `yt-automation-python-worker`

- [ ] Run the complete Python test suite and confirm zero failures.
- [ ] Run `git diff --check` and inspect the scoped diff.
- [ ] Rebuild the Python worker image through the existing Docker Compose configuration.
- [ ] Restart the Python worker and confirm its health endpoint responds successfully.
- [ ] Re-run the failed scene stage and verify existing scenes are reused and the workflow advances.
