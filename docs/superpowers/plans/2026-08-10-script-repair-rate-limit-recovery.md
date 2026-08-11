# Script Repair Rate-Limit Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve an already-valid script when a later duration-repair request is rate-limited, while returning HTTP 429 when no candidate was produced.

**Architecture:** Introduce a typed `ProviderRateLimitError` at the Groq boundary and map it to HTTP 429 at the FastAPI boundary. `LLMService` catches only that typed error inside its repair loop and returns its closest valid candidate when one exists; all other failures retain their current behavior.

**Tech Stack:** Python 3.11, `unittest`, FastAPI `TestClient`, Pydantic, httpx, Docker Compose, Groq OpenAI-compatible API

## Global Constraints

- Keep `qwen/qwen3.6-27b` as the primary story model and `llama-3.1-8b-instant` as the HTTP 429 fallback.
- Recover only from an exhausted HTTP 429; authentication, malformed-output, missing-content, and other integration errors remain fatal.
- Return the closest valid candidate when a later repair request is rate-limited.
- Return HTTP 429 when rate limiting occurs before any valid candidate exists.
- Do not change duration math, prompt content, retry counts, TTS, image generation, rendering, or backend BullMQ retry delays.
- Preserve unrelated changes already present in the dirty worktree.

---

### Task 1: Type exhausted Groq rate limits and expose HTTP 429

**Files:**
- Modify: `app/core/exceptions.py:18-31`
- Modify: `app/integrations/groq_client.py:6-100`
- Modify: `app/api/error_handlers.py:7-22`
- Modify: `tests/test_groq_client.py:87-240`
- Create: `tests/test_error_handlers.py`

**Interfaces:**
- Produces: `ProviderRateLimitError(message: str)` with `code == "provider_rate_limit"`
- Produces: `GroqClient.generate_text(...)` raising `ProviderRateLimitError` only when its final HTTP response is 429
- Produces: FastAPI JSON error response with HTTP 429 and `error.code == "provider_rate_limit"`

- [ ] **Step 1: Write failing provider and HTTP-boundary tests**

In `tests/test_groq_client.py`, import `ProviderRateLimitError` and add:

```python
def test_generate_text_raises_typed_error_when_primary_and_fallback_are_rate_limited(self) -> None:
    requested_models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_models.append(json.loads(request.content)["model"])
        return httpx.Response(
            429,
            json={"error": {"message": "token quota reached"}},
        )

    client = self.make_client(
        handler,
        fallback_model="llama-3.1-8b-instant",
    )

    with self.assertRaises(ProviderRateLimitError) as caught:
        client.generate_text(
            model="qwen/qwen3.6-27b",
            prompt="Return a story as JSON.",
        )

    self.assertEqual(
        requested_models,
        ["qwen/qwen3.6-27b", "llama-3.1-8b-instant"],
    )
    self.assertEqual(caught.exception.code, "provider_rate_limit")
    self.assertIn("status 429", str(caught.exception))
```

Create `tests/test_error_handlers.py`:

```python
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.error_handlers import register_exception_handlers
from app.core.exceptions import ProviderRateLimitError


class ErrorHandlerTests(unittest.TestCase):
    def test_provider_rate_limit_maps_to_http_429(self) -> None:
        app = FastAPI()
        register_exception_handlers(app)

        @app.get("/rate-limited")
        def rate_limited() -> None:
            raise ProviderRateLimitError("Groq quota reached")

        response = TestClient(app).get("/rate-limited")

        self.assertEqual(response.status_code, 429)
        self.assertEqual(
            response.json(),
            {
                "error": {
                    "code": "provider_rate_limit",
                    "message": "Groq quota reached",
                }
            },
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
python -m unittest tests.test_groq_client.GroqClientErrorTests.test_generate_text_raises_typed_error_when_primary_and_fallback_are_rate_limited tests.test_error_handlers -v
```

Expected: import/error failures because `ProviderRateLimitError` and its HTTP 429 mapping do not exist.

- [ ] **Step 3: Implement the typed exception and provider mapping**

In `app/core/exceptions.py`, allow integration subclasses to select a code and add the rate-limit type:

```python
class IntegrationError(AppError):
    def __init__(self, message: str, *, code: str = "integration_error") -> None:
        super().__init__(message, code=code)


class ProviderRateLimitError(IntegrationError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="provider_rate_limit")
```

In `GroqClient`, import `ProviderRateLimitError` and replace the terminal HTTP-error branch with:

```python
if response.status_code >= 400:
    message = self._error_message(response)
    if response.status_code == 429:
        raise ProviderRateLimitError(message)
    raise IntegrationError(message)
```

In `register_exception_handlers`, add the explicit rate-limit mapping before validation mapping:

```python
status_code = 500
if exc.code == "provider_rate_limit":
    status_code = 429
elif exc.code in {"validation_error", "media_error"}:
    status_code = 422
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```powershell
python -m unittest tests.test_groq_client tests.test_error_handlers -v
```

Expected: all Groq client and HTTP-boundary tests pass.

- [ ] **Step 5: Review the focused diff**

Run:

```powershell
git diff --check -- app/core/exceptions.py app/integrations/groq_client.py app/api/error_handlers.py tests/test_groq_client.py tests/test_error_handlers.py
git diff -- app/core/exceptions.py app/integrations/groq_client.py app/api/error_handlers.py tests/test_groq_client.py tests/test_error_handlers.py
```

Expected: only typed 429 classification, HTTP mapping, and their tests are added. Do not commit these implementation files separately because some contain pre-existing uncommitted work.

---

### Task 2: Recover the closest valid script from a later rate limit

**Files:**
- Modify: `app/services/llm_service.py:6-70`
- Modify: `tests/test_llm_service_duration.py:1-221`

**Interfaces:**
- Consumes: `ProviderRateLimitError` from Task 1
- Produces: `LLMService.generate_story_script(...)` returning `best_response` after a later typed 429 and re-raising when no candidate exists

- [ ] **Step 1: Write failing script-service recovery tests**

Import `ProviderRateLimitError` in `tests/test_llm_service_duration.py` and add:

```python
class RateLimitedAfterCandidateTextClient:
    def __init__(self) -> None:
        self.calls = 0

    def generate_text(self, **kwargs) -> str:
        self.calls += 1
        if self.calls == 1:
            return script_response(FAR_SHORT_NARRATION)
        raise ProviderRateLimitError("Groq quota reached")


class AlwaysRateLimitedTextClient:
    def generate_text(self, **kwargs) -> str:
        raise ProviderRateLimitError("Groq quota reached")
```

Add these tests to `LLMServiceDurationTests`:

```python
def test_existing_candidate_is_returned_when_duration_repair_is_rate_limited(self) -> None:
    client = RateLimitedAfterCandidateTextClient()
    service = LLMService(
        llm_client=client,
        model="qwen/qwen3.6-27b",
        allow_placeholder_generation=False,
    )
    payload = ScriptGenerationRequest(
        job_id="job-repair-rate-limit",
        project_id="project-repair-rate-limit",
        topic="The Mystery of the Laguna Copperplate Inscription",
        target_duration_seconds=60,
        speaking_rate=0.96,
        script_framework="history_story",
        story_format="mystery_reveal",
    )

    result = service.generate_story_script(payload)

    self.assertEqual(client.calls, 2)
    self.assertEqual(result.narration, FAR_SHORT_NARRATION)

def test_rate_limit_before_any_candidate_is_re_raised(self) -> None:
    service = LLMService(
        llm_client=AlwaysRateLimitedTextClient(),
        model="qwen/qwen3.6-27b",
        allow_placeholder_generation=False,
    )
    payload = ScriptGenerationRequest(
        job_id="job-initial-rate-limit",
        project_id="project-initial-rate-limit",
        topic="The Mystery of the Laguna Copperplate Inscription",
        target_duration_seconds=60,
        speaking_rate=0.96,
        script_framework="history_story",
        story_format="mystery_reveal",
    )

    with self.assertRaises(ProviderRateLimitError):
        service.generate_story_script(payload)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
python -m unittest tests.test_llm_service_duration.LLMServiceDurationTests.test_existing_candidate_is_returned_when_duration_repair_is_rate_limited tests.test_llm_service_duration.LLMServiceDurationTests.test_rate_limit_before_any_candidate_is_re_raised -v
```

Expected: the first test errors because the typed rate limit escapes despite an existing candidate; the second documents the already-required propagation behavior.

- [ ] **Step 3: Implement best-candidate recovery**

Import `ProviderRateLimitError` in `app/services/llm_service.py`. Wrap only the call to `self.llm_client.generate_text` inside the attempt loop:

```python
try:
    generated = self.llm_client.generate_text(
        model=self.model,
        prompt=self._build_prompt(
            payload,
            attempt=attempt,
            adjustment=adjustment,
            previous_response=last_response,
        ),
        max_new_tokens=2200,
        temperature=0.75,
    )
except ProviderRateLimitError:
    if best_response is not None:
        return best_response
    raise
```

Do not catch `IntegrationError` generally.

- [ ] **Step 4: Run all duration and provider-error tests**

Run:

```powershell
python -m unittest tests.test_llm_service_duration tests.test_llm_service_errors -v
```

Expected: all tests pass, including closest-duration behavior and provider-neutral non-rate-limit errors.

- [ ] **Step 5: Review the focused diff**

Run:

```powershell
git diff --check -- app/services/llm_service.py tests/test_llm_service_duration.py
git diff -- app/services/llm_service.py tests/test_llm_service_duration.py
```

Expected: only typed rate-limit recovery and its two regression tests are added beyond the existing dirty changes.

---

### Task 3: Verify and deploy the recovery behavior

**Files:**
- Verify: `app/core/exceptions.py`
- Verify: `app/integrations/groq_client.py`
- Verify: `app/api/error_handlers.py`
- Verify: `app/services/llm_service.py`
- Runtime: `docker-compose.yml`

**Interfaces:**
- Consumes: typed provider rate limits and best-candidate recovery from Tasks 1 and 2
- Produces: a rebuilt `yt-automation-python-worker` that never converts a Groq-only 429 into an unrecoverable duration-repair HTTP 500

- [ ] **Step 1: Run the complete Python test suite**

Run:

```powershell
python -m unittest discover -s tests -p 'test_*.py' -v
```

Expected: all tests pass with zero failures or errors.

- [ ] **Step 2: Compile and check the worktree**

Run:

```powershell
python -m compileall -q app tests
git diff --check
```

Expected: exit code 0 and no compilation or whitespace errors.

- [ ] **Step 3: Rebuild only the Python worker**

Run:

```powershell
docker compose up -d --build --force-recreate python-worker
```

Expected: the image builds and `yt-automation-python-worker` starts.

- [ ] **Step 4: Verify runtime configuration without printing secrets**

Run:

```powershell
docker exec yt-automation-python-worker python -c "from app.core.config import get_settings; s=get_settings(); assert s.groq_api_key and s.gemini_api_key; assert s.groq_model == 'qwen/qwen3.6-27b'; assert s.groq_fallback_model == 'llama-3.1-8b-instant'; print('runtime_config=ok')"
```

Expected: `runtime_config=ok`.

- [ ] **Step 5: Verify the real script endpoint**

POST the failed project's Laguna Copperplate settings to `/internal/faceless/generate-script` with a temporary job and project ID. Report only HTTP status, word count, estimated duration, and scene count.

Expected outcomes:

- HTTP 200 when either model produces at least one valid candidate, even if a later repair call is rate-limited; or
- HTTP 429 when both models are rate-limited before any valid candidate exists.

HTTP 500 with a Groq rate-limit message is a failure.

- [ ] **Step 6: Inspect timestamped logs**

Run:

```powershell
docker logs --timestamps --since 10m yt-automation-python-worker
```

Expected: the live request terminates with `200 OK` or `429 Too Many Requests`, never a rate-limit-driven `500 Internal Server Error`.

- [ ] **Step 7: Preserve the branch and worktree**

Do not merge, push, reset, clean, or commit implementation files. The user previously selected keeping branch `LOGIC_CHANGE` as-is, and the repository contains unrelated changes.
