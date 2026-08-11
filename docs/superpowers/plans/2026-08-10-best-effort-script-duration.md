# Best-Effort Script Duration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve duration-repair attempts while returning the closest valid script instead of HTTP 500 when every candidate misses the preferred duration window.

**Architecture:** `LLMService` remains responsible for parsing candidates, estimating duration, and building repair prompts. During its existing three-attempt loop it will track the valid candidate with the smallest absolute duration error, return immediately for an in-window candidate, and otherwise return the closest candidate after the final attempt. Provider, JSON, and content errors remain fatal.

**Tech Stack:** Python 3.11, `unittest`, Pydantic, FastAPI, Docker Compose, Groq OpenAI-compatible API

## Global Constraints

- Keep `qwen/qwen3.6-27b` as the primary story model and `llama-3.1-8b-instant` as the HTTP 429 fallback.
- Keep the preferred duration window at 90% through 108% of the requested target.
- Allow a valid script shorter or longer than 60 seconds when it is the closest candidate after three attempts.
- Do not change TTS, image generation, rendering, speaking-rate math, or queue retry policy.
- Preserve unrelated changes already present in the dirty worktree.

---

### Task 1: Return the closest valid script after duration repair is exhausted

**Files:**
- Modify: `tests/test_llm_service_duration.py:8-174`
- Modify: `app/services/llm_service.py:33-60`

**Interfaces:**
- Consumes: `LLMService._estimate_narration_duration_seconds(narration: str, speaking_rate: float) -> float`
- Produces: `LLMService.generate_story_script(payload: ScriptGenerationRequest) -> ScriptGenerationResponse` returning the closest parsed candidate after three out-of-window attempts

- [ ] **Step 1: Write the failing regression test**

Add these fixtures and client beside the existing duration-test fixtures:

```python
FAR_SHORT_NARRATION = f"{HOOK} {' '.join(['detail'] * 50)}"
FAR_LONG_NARRATION = f"{HOOK} {' '.join(['detail'] * 150)}"
CLOSEST_LONG_NARRATION = f"{HOOK} {' '.join(['detail'] * 140)}"


class AlwaysOutOfWindowTextClient:
    def __init__(self) -> None:
        self.calls = 0
        self.responses = (
            FAR_SHORT_NARRATION,
            FAR_LONG_NARRATION,
            CLOSEST_LONG_NARRATION,
        )

    def generate_text(self, **kwargs) -> str:
        narration = self.responses[self.calls]
        self.calls += 1
        return script_response(narration)
```

Add this test to `LLMServiceDurationTests`:

```python
def test_closest_valid_script_is_returned_after_duration_retries(self) -> None:
    client = AlwaysOutOfWindowTextClient()
    service = LLMService(
        llm_client=client,
        model="qwen/qwen3.6-27b",
        allow_placeholder_generation=False,
    )
    payload = ScriptGenerationRequest(
        job_id="job-best-effort-duration",
        project_id="project-best-effort-duration",
        topic="The Mystery of the Laguna Copperplate Inscription",
        target_duration_seconds=60,
        speaking_rate=0.96,
        script_framework="history_story",
        story_format="mystery_reveal",
    )

    result = service.generate_story_script(payload)

    self.assertEqual(client.calls, 3)
    self.assertEqual(result.narration, CLOSEST_LONG_NARRATION)
    self.assertGreater(
        service._estimate_narration_duration_seconds(
            result.narration,
            payload.speaking_rate,
        ),
        payload.target_duration_seconds * service.MAX_DURATION_RATIO,
    )
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
python -m unittest tests.test_llm_service_duration.LLMServiceDurationTests.test_closest_valid_script_is_returned_after_duration_retries -v
```

Expected: FAIL because `generate_story_script` raises `IntegrationError` after the third valid candidate remains outside the preferred duration window.

- [ ] **Step 3: Implement closest-candidate tracking**

In `generate_story_script`, keep `last_response` for the repair prompt and add `best_response` plus `best_duration_error`. After parsing each candidate, compute its estimated duration and absolute difference from the requested target. Replace the best candidate only when the new difference is smaller. Preserve the immediate return for an in-window candidate. Replace the terminal duration-only `IntegrationError` with `return best_response`:

```python
last_response: ScriptGenerationResponse | None = None
best_response: ScriptGenerationResponse | None = None
best_duration_error = float("inf")
adjustment = "expand"

for attempt in range(1, self.SCRIPT_MAX_ATTEMPTS + 1):
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
    candidate = self._parse_response(payload, generated)
    estimated_duration = self._estimate_narration_duration_seconds(
        candidate.narration,
        payload.speaking_rate,
    )
    duration_error = abs(estimated_duration - payload.target_duration_seconds)
    if duration_error < best_duration_error:
        best_response = candidate
        best_duration_error = duration_error
    if self._meets_duration_target(payload, candidate):
        return candidate
    last_response = candidate
    adjustment = (
        "shorten"
        if estimated_duration > payload.target_duration_seconds
        else "expand"
    )

if best_response is not None:
    return best_response
```

Keep `raise IntegrationError("The LLM did not return a usable script.")` for the no-valid-candidate path.

- [ ] **Step 4: Run focused duration tests and verify GREEN**

Run:

```powershell
python -m unittest tests.test_llm_service_duration -v
```

Expected: all duration tests pass, including the new three-attempt best-effort case and the existing repair cases.

- [ ] **Step 5: Review the focused diff**

Run:

```powershell
git diff -- app/services/llm_service.py tests/test_llm_service_duration.py
git diff --check -- app/services/llm_service.py tests/test_llm_service_duration.py
```

Expected: only closest-candidate behavior and its regression test are added. Do not commit these files separately because they contain pre-existing uncommitted work that must remain under the user's control.

---

### Task 2: Verify and deploy the Python worker

**Files:**
- Verify: `app/services/llm_service.py`
- Verify: `tests/test_llm_service_duration.py`
- Runtime: `docker-compose.yml`

**Interfaces:**
- Consumes: the updated `LLMService.generate_story_script` behavior from Task 1
- Produces: a rebuilt `yt-automation-python-worker` whose `/internal/faceless/generate-script` endpoint returns a valid script instead of duration-only HTTP 500

- [ ] **Step 1: Run the complete Python test suite**

Run:

```powershell
python -m unittest discover -s tests -p 'test_*.py' -v
```

Expected: all tests pass with zero failures or errors.

- [ ] **Step 2: Compile all changed Python sources**

Run:

```powershell
python -m compileall -q app tests
```

Expected: exit code 0 and no compilation errors.

- [ ] **Step 3: Rebuild and recreate only the Python worker**

Run:

```powershell
docker compose up -d --build --force-recreate python-worker
```

Expected: image build completes and `yt-automation-python-worker` starts.

- [ ] **Step 4: Verify runtime model and credentials without exposing secrets**

Run:

```powershell
docker exec yt-automation-python-worker python -c "from app.core.config import get_settings; s=get_settings(); assert s.groq_api_key; assert s.gemini_api_key; print('story_model='+s.groq_model); print('fallback_model='+str(s.groq_fallback_model)); print('keys_configured=True')"
```

Expected:

```text
story_model=qwen/qwen3.6-27b
fallback_model=llama-3.1-8b-instant
keys_configured=True
```

- [ ] **Step 5: Verify the real script endpoint with the failed project's settings**

Run a POST to `http://127.0.0.1:8000/internal/faceless/generate-script` using a temporary job/project ID and these settings: topic `The Mystery of the Laguna Copperplate Inscription`, target duration `60`, speaking rate `0.96`, framework `history_story`, and format `mystery_reveal`.

Expected: HTTP 200 with a non-empty title, narration, and scenes. Report only response status, narration word count, estimated duration, and scene count; do not print the generated script body.

- [ ] **Step 6: Inspect fresh service logs**

Run:

```powershell
docker logs --since 10m yt-automation-python-worker
```

Expected: the verification request ends with `POST /internal/faceless/generate-script HTTP/1.1` `200 OK`. A Qwen 429 followed by a Llama 200 is acceptable fallback behavior.

- [ ] **Step 7: Preserve the branch and working tree**

Do not merge, push, reset, clean, or commit the implementation files. The user previously selected keeping branch `LOGIC_CHANGE` as-is, and the repository contains unrelated changes.
