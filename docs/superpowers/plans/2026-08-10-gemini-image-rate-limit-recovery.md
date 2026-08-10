# Gemini Image Rate-Limit Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve Gemini HTTP 429 responses as typed provider-rate-limit failures so the backend queues scene-image generation for retry instead of marking it as an HTTP 500 failure.

**Architecture:** `GeminiClient` remains the provider boundary and classifies only HTTP 429 as `ProviderRateLimitError`. The existing FastAPI exception handler maps that exception to HTTP 429, and the existing backend worker applies durable BullMQ backoff. Other Gemini errors remain `IntegrationError`.

**Tech Stack:** Python 3.11, httpx, unittest, FastAPI, Docker Compose

## Global Constraints

- Do not change the Gemini model, API key, prompt, image dimensions, or backend retry timings.
- Preserve non-429 Gemini error behavior.
- Rebuild and recreate only the Python worker after the full test suite passes.

---

### Task 1: Classify Gemini HTTP 429 responses

**Files:**
- Create: `tests/test_gemini_client.py`
- Modify: `app/integrations/gemini_client.py:7-9,114-121`

**Interfaces:**
- Consumes: `ProviderRateLimitError(message: str)` from `app.core.exceptions`.
- Produces: `GeminiClient.generate_image(...)` raises `ProviderRateLimitError` for provider HTTP 429 and `IntegrationError` for other HTTP failures.

- [x] **Step 1: Write the failing tests**

```python
import unittest
from unittest.mock import patch

import httpx

from app.core.exceptions import IntegrationError, ProviderRateLimitError
from app.integrations.gemini_client import GeminiClient


class GeminiClientErrorTests(unittest.TestCase):
    def make_client(self) -> GeminiClient:
        return GeminiClient(
            api_key="gemini-test-key",
            base_url="https://generativelanguage.test/v1beta/interactions",
            timeout_seconds=30,
        )

    def test_generate_image_raises_typed_error_for_rate_limit(self) -> None:
        response = httpx.Response(
            429,
            json={"error": {"code": "rate_limit_exceeded", "message": "image quota reached"}},
        )
        with patch("app.integrations.gemini_client.httpx.Client") as client_type:
            client_type.return_value.__enter__.return_value.post.return_value = response
            with self.assertRaises(ProviderRateLimitError) as caught:
                self.make_client().generate_image(model="gemini-image-model", prompt="scene")

        self.assertEqual(caught.exception.code, "provider_rate_limit")
        self.assertIn("status 429", str(caught.exception))
        self.assertIn("image quota reached", str(caught.exception))

    def test_generate_image_keeps_non_rate_limit_as_integration_error(self) -> None:
        response = httpx.Response(
            403,
            json={"error": {"code": "permission_denied", "message": "permission denied"}},
        )
        with patch("app.integrations.gemini_client.httpx.Client") as client_type:
            client_type.return_value.__enter__.return_value.post.return_value = response
            with self.assertRaises(IntegrationError) as caught:
                self.make_client().generate_image(model="gemini-image-model", prompt="scene")

        self.assertNotIsInstance(caught.exception, ProviderRateLimitError)
        self.assertIn("status 403", str(caught.exception))
```

- [x] **Step 2: Run the focused tests and verify the 429 test fails**

Run: `python -m unittest tests.test_gemini_client -v`

Expected: the 429 case raises `IntegrationError`, so `assertRaises(ProviderRateLimitError)` fails; the 403 case passes.

- [x] **Step 3: Implement the minimal provider classification**

Update the exceptions import:

```python
from app.core.exceptions import IntegrationError, ProviderRateLimitError
```

Update `_post_interaction`:

```python
        if response.status_code >= 400:
            message = self._error_message(response, prefix=error_prefix)
            if response.status_code == 429:
                raise ProviderRateLimitError(message)
            raise IntegrationError(message)
```

- [x] **Step 4: Run focused tests**

Run: `python -m unittest tests.test_gemini_client tests.test_error_handlers -v`

Expected: all Gemini classification and FastAPI mapping tests pass.

- [x] **Step 5: Run the complete Python test suite**

Run: `python -m unittest discover -s tests -v`

Expected: all tests pass.

- [x] **Step 6: Rebuild and recreate the Python worker**

Run from `yt_automation`: `docker compose up -d --build python-worker`

Expected: `yt-automation-python-worker` is recreated and healthy/running; backend containers are not rebuilt.

- [x] **Step 7: Verify deployed code and runtime health**

Run: `docker compose ps python-worker`

Run: `docker exec yt-automation-python-worker python -c "from app.core.exceptions import ProviderRateLimitError; from app.integrations.gemini_client import GeminiClient; print('gemini-rate-limit-classification-ready')"`

Expected: the worker is running and prints `gemini-rate-limit-classification-ready`.
