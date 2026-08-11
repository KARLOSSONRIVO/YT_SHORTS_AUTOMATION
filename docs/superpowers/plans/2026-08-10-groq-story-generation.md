# Groq Story Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate story scripts with Groq while Gemini continues to generate scene images and narration audio.

**Architecture:** Add a focused HTTP-based `GroqClient` that implements the existing `generate_text` call shape consumed by `LLMService`. Configure and inject that client only at the script-generation dependency boundary; leave Gemini image, TTS, and video dependencies unchanged.

**Tech Stack:** Python 3.11+, `httpx`, Pydantic Settings, FastAPI dependency functions, standard-library `unittest`

## Global Constraints

- Groq is required for story generation; there is no Gemini fallback.
- Default story model: `llama-3.3-70b-versatile`.
- Gemini remains responsible for image generation and text-to-speech.
- Use the existing `httpx` dependency; do not add Groq or OpenAI SDK packages.
- Do not change public API request or response schemas.
- Do not change script prompts, duration tuning, scene parsing, image generation, TTS behavior, or rendering behavior.
- Preserve every pre-existing uncommitted change, especially in `app/core/config.py`, `app/api/deps.py`, and `app/services/llm_service.py`.
- Do not stage or commit pre-existing user changes. The implementation tasks that overlap dirty files intentionally omit commit commands.
- Automated tests must not make live Groq or Gemini requests.

---

## File Structure

- Create `app/integrations/groq_client.py`: Groq authentication, request serialization, response extraction, and provider-specific transport errors.
- Create `tests/__init__.py`: marks the test directory as an importable package.
- Create `tests/test_groq_client.py`: request-contract and failure-path tests using `httpx.MockTransport`.
- Create `tests/test_provider_routing.py`: environment-setting and dependency-injection routing tests.
- Create `tests/test_llm_service_errors.py`: provider-neutral service error regression test.
- Modify `app/core/config.py`: Groq key, model, and base URL settings.
- Modify `app/api/deps.py`: cached Groq client and Groq-backed `LLMService` wiring.
- Modify `app/services/llm_service.py`: replace the stale Gemini-specific wrapper message.
- Modify `README.md`: document provider responsibilities and required environment keys.

### Task 1: Groq Successful Request Contract

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_groq_client.py`
- Create: `app/integrations/groq_client.py`

**Interfaces:**
- Consumes: `httpx.BaseTransport`, `app.core.exceptions.IntegrationError`
- Produces: `GroqClient(api_key: str | None, base_url: str, timeout_seconds: float | None, transport: httpx.BaseTransport | None = None)`
- Produces: `GroqClient.generate_text(*, model: str, prompt: str, max_new_tokens: int = 1600, temperature: float = 0.7) -> str`

- [ ] **Step 1: Write the failing successful-request test**

Create an empty `tests/__init__.py`, then create `tests/test_groq_client.py`:

```python
import json
import unittest

import httpx

from app.integrations.groq_client import GroqClient


class GroqClientRequestTests(unittest.TestCase):
    def test_generate_text_sends_chat_completion_request_and_returns_content(self) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["authorization"] = request.headers.get("Authorization")
            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": '{"title":"A strong hook"}'}}
                    ]
                },
            )

        client = GroqClient(
            api_key="groq-test-key",
            base_url="https://api.groq.test/openai/v1/",
            timeout_seconds=30,
            transport=httpx.MockTransport(handler),
        )

        result = client.generate_text(
            model="llama-3.3-70b-versatile",
            prompt="Return a story as JSON.",
            max_new_tokens=2200,
            temperature=0.75,
        )

        self.assertEqual(result, '{"title":"A strong hook"}')
        self.assertEqual(captured["url"], "https://api.groq.test/openai/v1/chat/completions")
        self.assertEqual(captured["authorization"], "Bearer groq-test-key")
        self.assertEqual(
            captured["payload"],
            {
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": "Return a story as JSON."}],
                "temperature": 0.75,
                "max_completion_tokens": 2200,
                "response_format": {"type": "json_object"},
            },
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
python -m unittest tests.test_groq_client.GroqClientRequestTests.test_generate_text_sends_chat_completion_request_and_returns_content -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'app.integrations.groq_client'`.

- [ ] **Step 3: Implement the minimal successful Groq request**

Create `app/integrations/groq_client.py`:

```python
from __future__ import annotations

import httpx


class GroqClient:
    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str,
        timeout_seconds: float | None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds if timeout_seconds and timeout_seconds > 0 else None
        self.transport = transport

    def generate_text(
        self,
        *,
        model: str,
        prompt: str,
        max_new_tokens: int = 1600,
        temperature: float = 0.7,
    ) -> str:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_completion_tokens": max_new_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key or ''}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.timeout_seconds, transport=self.transport) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
        body = response.json()
        return str(body["choices"][0]["message"]["content"])
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```powershell
python -m unittest tests.test_groq_client.GroqClientRequestTests.test_generate_text_sends_chat_completion_request_and_returns_content -v
```

Expected: one test passes.

- [ ] **Step 5: Commit the new isolated client contract**

```powershell
git add -- app/integrations/groq_client.py tests/__init__.py tests/test_groq_client.py
git diff --cached --check
git commit -m "feat: add Groq text generation client"
```

### Task 2: Groq Validation and Error Handling

**Files:**
- Modify: `tests/test_groq_client.py`
- Modify: `app/integrations/groq_client.py`

**Interfaces:**
- Consumes: `GroqClient.generate_text(...) -> str` from Task 1
- Produces: `GroqClient.is_configured() -> bool`
- Produces: `IntegrationError` for missing configuration, HTTP failures, non-JSON responses, and missing assistant content

- [ ] **Step 1: Add failing validation and response-error tests**

Add these imports to `tests/test_groq_client.py`:

```python
from app.core.exceptions import IntegrationError
```

Add this test class before the `if __name__ == "__main__"` block:

```python
class GroqClientErrorTests(unittest.TestCase):
    def make_client(
        self,
        handler,
        *,
        api_key: str | None = "groq-test-key",
    ) -> GroqClient:
        return GroqClient(
            api_key=api_key,
            base_url="https://api.groq.test/openai/v1",
            timeout_seconds=30,
            transport=httpx.MockTransport(handler),
        )

    def test_is_configured_requires_an_api_key(self) -> None:
        self.assertTrue(self.make_client(lambda request: httpx.Response(200), api_key="key").is_configured())
        self.assertFalse(self.make_client(lambda request: httpx.Response(200), api_key=None).is_configured())

    def test_generate_text_requires_api_key(self) -> None:
        client = self.make_client(lambda request: httpx.Response(200), api_key=None)
        with self.assertRaisesRegex(IntegrationError, "GROQ_API_KEY is required"):
            client.generate_text(model="llama-3.3-70b-versatile", prompt="story")

    def test_generate_text_requires_model(self) -> None:
        client = self.make_client(lambda request: httpx.Response(200))
        with self.assertRaisesRegex(IntegrationError, "GROQ_MODEL is required"):
            client.generate_text(model="", prompt="story")

    def test_generate_text_reports_http_errors(self) -> None:
        client = self.make_client(
            lambda request: httpx.Response(
                401,
                json={"error": {"message": "invalid API key"}},
            )
        )
        with self.assertRaisesRegex(
            IntegrationError,
            "Groq text API failed with status 401: invalid API key",
        ):
            client.generate_text(model="llama-3.3-70b-versatile", prompt="story")

    def test_generate_text_rejects_non_json_response(self) -> None:
        client = self.make_client(
            lambda request: httpx.Response(200, text="not-json")
        )
        with self.assertRaisesRegex(IntegrationError, "Groq returned a non-JSON response"):
            client.generate_text(model="llama-3.3-70b-versatile", prompt="story")

    def test_generate_text_requires_assistant_content(self) -> None:
        client = self.make_client(
            lambda request: httpx.Response(200, json={"choices": []})
        )
        with self.assertRaisesRegex(IntegrationError, "returned no text output"):
            client.generate_text(model="llama-3.3-70b-versatile", prompt="story")
```

- [ ] **Step 2: Run the error tests and verify RED**

Run:

```powershell
python -m unittest tests.test_groq_client.GroqClientErrorTests -v
```

Expected: tests fail because `is_configured` and the explicit `IntegrationError` paths do not exist.

- [ ] **Step 3: Implement configuration validation and safe response parsing**

Replace `app/integrations/groq_client.py` with:

```python
from __future__ import annotations

import json
from typing import Any

import httpx

from app.core.exceptions import IntegrationError


class GroqClient:
    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str,
        timeout_seconds: float | None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds if timeout_seconds and timeout_seconds > 0 else None
        self.transport = transport

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def generate_text(
        self,
        *,
        model: str,
        prompt: str,
        max_new_tokens: int = 1600,
        temperature: float = 0.7,
    ) -> str:
        if not self.api_key:
            raise IntegrationError("GROQ_API_KEY is required for Groq story generation.")
        if not model:
            raise IntegrationError("GROQ_MODEL is required for Groq story generation.")

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_completion_tokens": max_new_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.timeout_seconds, transport=self.transport) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )

        if response.status_code >= 400:
            raise IntegrationError(self._error_message(response))

        try:
            body = response.json()
        except json.JSONDecodeError as exc:
            raise IntegrationError("Groq text API failed: Groq returned a non-JSON response.") from exc

        content = self._extract_content(body)
        if not content:
            raise IntegrationError(f"Groq text API returned no text output: {body}")
        return content

    def _extract_content(self, body: Any) -> str | None:
        if not isinstance(body, dict):
            return None
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        first = choices[0]
        if not isinstance(first, dict):
            return None
        message = first.get("message")
        if not isinstance(message, dict):
            return None
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
        return None

    def _error_message(self, response: httpx.Response) -> str:
        try:
            body = response.json()
        except json.JSONDecodeError:
            body = response.text
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                message = error.get("message") or error
                return f"Groq text API failed with status {response.status_code}: {message}"
            if error:
                return f"Groq text API failed with status {response.status_code}: {error}"
        return f"Groq text API failed with status {response.status_code}: {body}"
```

- [ ] **Step 4: Run all Groq client tests and verify GREEN**

Run:

```powershell
python -m unittest tests.test_groq_client -v
```

Expected: all Groq client tests pass without network access.

- [ ] **Step 5: Commit the hardened client**

```powershell
git add -- app/integrations/groq_client.py tests/test_groq_client.py
git diff --cached --check
git commit -m "test: harden Groq client failures"
```

### Task 3: Configure Groq and Route Only Scripts to It

**Files:**
- Create: `tests/test_provider_routing.py`
- Modify: `app/core/config.py:21-36`
- Modify: `app/api/deps.py:3-7,52-58,145-163`

**Interfaces:**
- Consumes: `GroqClient` from Tasks 1-2
- Produces: `Settings.groq_api_key: str | None`
- Produces: `Settings.groq_model: str`
- Produces: `Settings.groq_base_url: str`
- Produces: `get_groq_client() -> GroqClient`
- Preserves: `get_gemini_image_generation_service()` and `get_tts_service()` Gemini injection

- [ ] **Step 1: Write failing settings and provider-routing tests**

Create `tests/test_provider_routing.py`:

```python
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.api import deps
from app.core.config import Settings


class GroqSettingsTests(unittest.TestCase):
    def test_settings_load_bare_groq_key_and_default_story_model(self) -> None:
        with patch.dict(os.environ, {"GROQ_API_KEY": "groq-env-key"}, clear=True):
            settings = Settings(_env_file=None)

        self.assertTrue(hasattr(settings, "groq_api_key"))
        self.assertEqual(settings.groq_api_key, "groq-env-key")
        self.assertEqual(settings.groq_model, "llama-3.3-70b-versatile")
        self.assertEqual(settings.groq_base_url, "https://api.groq.com/openai/v1")


class ProviderRoutingTests(unittest.TestCase):
    def tearDown(self) -> None:
        deps.get_llm_service.cache_clear()
        deps.get_gemini_image_generation_service.cache_clear()
        deps.get_tts_service.cache_clear()

    def test_llm_service_receives_groq_client_and_model(self) -> None:
        settings = SimpleNamespace(
            groq_model="llama-3.3-70b-versatile",
            gemini_model="gemini-story-model",
            allow_placeholder_generation=False,
        )
        groq_client = object()
        gemini_client = object()
        deps.get_llm_service.cache_clear()

        with (
            patch.object(deps, "get_settings", return_value=settings),
            patch.object(deps, "get_groq_client", return_value=groq_client, create=True),
            patch.object(deps, "get_gemini_client", return_value=gemini_client),
        ):
            service = deps.get_llm_service()

        self.assertIs(service.llm_client, groq_client)
        self.assertEqual(service.model, "llama-3.3-70b-versatile")

    def test_image_generation_service_still_receives_gemini_client(self) -> None:
        settings = SimpleNamespace(gemini_image_model="gemini-image-model")
        gemini_client = object()
        deps.get_gemini_image_generation_service.cache_clear()

        with (
            patch.object(deps, "get_settings", return_value=settings),
            patch.object(deps, "get_gemini_client", return_value=gemini_client),
        ):
            service = deps.get_gemini_image_generation_service()

        self.assertIs(service.gemini_client, gemini_client)
        self.assertEqual(service.model, "gemini-image-model")

    def test_tts_service_still_receives_gemini_client(self) -> None:
        settings = SimpleNamespace(
            output_dir="outputs",
            gemini_tts_model="gemini-tts-model",
            allow_placeholder_generation=False,
        )
        gemini_client = object()
        ffmpeg_client = object()
        deps.get_tts_service.cache_clear()

        with (
            patch.object(deps, "get_settings", return_value=settings),
            patch.object(deps, "get_gemini_client", return_value=gemini_client),
            patch.object(deps, "get_ffmpeg_client", return_value=ffmpeg_client),
        ):
            service = deps.get_tts_service()

        self.assertIs(service.gemini_client, gemini_client)
        self.assertIs(service.ffmpeg_client, ffmpeg_client)
        self.assertEqual(service.model, "gemini-tts-model")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the provider-routing tests and verify RED**

Run:

```powershell
python -m unittest tests.test_provider_routing -v
```

Expected: FAIL because Groq settings, `get_groq_client`, and Groq-backed LLM wiring do not exist.

- [ ] **Step 3: Add Groq settings without disturbing Gemini configuration**

In `app/core/config.py`, replace the stale provider comment and add these fields immediately before the Gemini fields:

```python
    # Groq generates story scripts. Gemini generates images, speech, and video.
    groq_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("PY_WORKER_GROQ_API_KEY", "GROQ_API_KEY"),
    )
    groq_model: str = Field(
        default="llama-3.3-70b-versatile",
        validation_alias=AliasChoices("PY_WORKER_GROQ_MODEL", "GROQ_MODEL"),
    )
    groq_base_url: str = Field(
        default="https://api.groq.com/openai/v1",
        validation_alias=AliasChoices("PY_WORKER_GROQ_BASE_URL", "GROQ_BASE_URL"),
    )
```

Keep every existing Gemini field and the user's unrelated settings edits intact.

- [ ] **Step 4: Add Groq dependency wiring and preserve Gemini media wiring**

In `app/api/deps.py`, add the import:

```python
from app.integrations.groq_client import GroqClient
```

Add this cached dependency immediately after `get_gemini_client`:

```python
@lru_cache
def get_groq_client() -> GroqClient:
    settings = get_settings()
    return GroqClient(
        api_key=settings.groq_api_key,
        base_url=settings.groq_base_url,
        timeout_seconds=settings.ai_timeout_seconds,
    )
```

Change only `get_llm_service` to:

```python
@lru_cache
def get_llm_service() -> LLMService:
    settings = get_settings()
    return LLMService(
        llm_client=get_groq_client(),
        model=settings.groq_model,
        allow_placeholder_generation=settings.allow_placeholder_generation,
    )
```

Do not change `get_gemini_image_generation_service`, `get_tts_service`, or `get_ai_animation_service`.

- [ ] **Step 5: Run the provider-routing tests and verify GREEN**

Run:

```powershell
python -m unittest tests.test_provider_routing -v
```

Expected: four tests pass, proving Groq is used for scripts and Gemini remains wired to images and TTS.

- [ ] **Step 6: Review the dirty-file delta without staging user work**

Run:

```powershell
git diff -- app/core/config.py app/api/deps.py tests/test_provider_routing.py
git diff --check
```

Confirm that the earlier removals of Reddit background settings and all other pre-existing edits remain unchanged. Do not stage or commit these overlapping files in this task.

### Task 4: Provider-Neutral Script Errors and Operator Documentation

**Files:**
- Create: `tests/test_llm_service_errors.py`
- Modify: `app/services/llm_service.py:57-62`
- Modify: `README.md`

**Interfaces:**
- Consumes: existing `LLMService.generate_story_script(payload: ScriptGenerationRequest) -> ScriptGenerationResponse`
- Produces: provider-neutral `IntegrationError("Story script generation failed: ...")` for unexpected client failures
- Documents: `GROQ_API_KEY` for stories and `GEMINI_API_KEY` for image/TTS stages

- [ ] **Step 1: Write the failing provider-neutral error test**

Create `tests/test_llm_service_errors.py`:

```python
import unittest

from app.core.exceptions import IntegrationError
from app.schemas.faceless_video import ScriptGenerationRequest
from app.services.llm_service import LLMService


class FailingTextClient:
    def generate_text(self, **kwargs) -> str:
        raise RuntimeError("provider offline")


class LLMServiceErrorTests(unittest.TestCase):
    def test_unexpected_client_failure_uses_provider_neutral_message(self) -> None:
        service = LLMService(
            llm_client=FailingTextClient(),
            model="llama-3.3-70b-versatile",
            allow_placeholder_generation=False,
        )
        payload = ScriptGenerationRequest(
            job_id="job-1",
            project_id="project-1",
            topic="A forgotten historical event",
        )

        with self.assertRaisesRegex(
            IntegrationError,
            "Story script generation failed: provider offline",
        ):
            service.generate_story_script(payload)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the focused error test and verify RED**

Run:

```powershell
python -m unittest tests.test_llm_service_errors -v
```

Expected: FAIL because the current message says `Gemini script generation failed`.

- [ ] **Step 3: Make the minimal provider-neutral error change**

In `app/services/llm_service.py`, replace only the final wrapper line:

```python
            raise IntegrationError(f"Story script generation failed: {exc}") from exc
```

Do not alter the user's duration-ratio, retry-adjustment, story-format, or speaking-rate changes.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```powershell
python -m unittest tests.test_llm_service_errors -v
```

Expected: one test passes.

- [ ] **Step 5: Document the provider split**

Add this section after Quick Start in `README.md`:

````markdown
## AI providers

Story scripts use Groq. Scene images and narration speech use Gemini.

Set these values in `.env`:

```text
GROQ_API_KEY=your-groq-api-key
GROQ_MODEL=llama-3.3-70b-versatile
GEMINI_API_KEY=your-gemini-api-key
```

`GROQ_MODEL` is optional and defaults to `llama-3.3-70b-versatile`. Script generation does not fall back to Gemini when Groq is unavailable.
````

- [ ] **Step 6: Review the dirty-file delta without staging user work**

Run:

```powershell
git diff -- app/services/llm_service.py README.md tests/test_llm_service_errors.py
git diff --check
```

Confirm that only the provider error label changed in `LLMService` and that all pre-existing script-timing edits remain intact. Do not stage or commit the overlapping `LLMService` file.

### Task 5: Full Verification

**Files:**
- Verify: `app/integrations/groq_client.py`
- Verify: `app/core/config.py`
- Verify: `app/api/deps.py`
- Verify: `app/services/llm_service.py`
- Verify: `README.md`
- Verify: `tests/`

**Interfaces:**
- Consumes: all implementation outputs from Tasks 1-4
- Produces: fresh evidence that imports compile, all tests pass, the application loads, and the diff has no whitespace errors

- [ ] **Step 1: Run the full unit-test suite**

```powershell
python -m unittest discover -s tests -v
```

Expected: every test passes with no live provider requests.

- [ ] **Step 2: Compile application and test modules**

```powershell
python -m compileall -q app tests
```

Expected: exit code 0 with no syntax errors.

- [ ] **Step 3: Verify application import and route construction**

```powershell
python -c "from app.main import app; print(app.title)"
```

Expected: exit code 0 and the FastAPI application title.

- [ ] **Step 4: Check the complete working-tree diff**

```powershell
git diff --check
git status --short
git diff -- app/integrations/groq_client.py app/core/config.py app/api/deps.py app/services/llm_service.py README.md tests
```

Expected: no whitespace errors; Groq appears only in script configuration/client wiring; Gemini remains in image and TTS wiring; all pre-existing user changes remain present.

- [ ] **Step 5: Report the commit boundary honestly**

Report the test counts and command exit codes. List the Groq-client commits created in Tasks 1-2, then state that changes overlapping the user's dirty files were deliberately left uncommitted so no pre-existing work was staged without authorization.
