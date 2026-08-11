# Groq Model Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Qwen 3.6 27B the primary script model, fall back once to Llama 3.1 8B on Groq HTTP 429 responses, and explicitly keep topic research on Compound Mini.

**Architecture:** Add the fallback model to Python worker settings and inject it into `GroqClient`. `GroqClient.generate_text` owns the HTTP-status decision because it can distinguish a 429 from unrelated failures; it resends the identical payload with only the model changed. Backend research remains independently configured through `GROQ_TOPIC_RESEARCH_MODEL`.

**Tech Stack:** Python 3, Pydantic Settings, httpx, unittest, dotenv, Docker Compose, TypeScript backend configuration

## Global Constraints

- Primary script model: `qwen/qwen3.6-27b`.
- Rate-limit fallback model: `llama-3.1-8b-instant`.
- Research model: `groq/compound-mini`.
- Retry the fallback exactly once and only after HTTP 429.
- Never route text generation back to Gemini.

---

### Task 1: Configuration and Dependency Wiring

**Files:**
- Modify: `app/core/config.py`
- Modify: `app/api/deps.py`
- Test: `tests/test_provider_routing.py`

**Interfaces:**
- Consumes: `Settings` environment aliases.
- Produces: `Settings.groq_fallback_model: str | None` and `GroqClient(..., fallback_model=...)` wiring.

- [ ] **Step 1: Write failing configuration and wiring tests**

Update tests to expect `qwen/qwen3.6-27b`, `llama-3.1-8b-instant`, and the fallback argument passed into the client.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_provider_routing -v`

Expected: failures because the primary default is still Llama 3.3 and no fallback setting exists.

- [ ] **Step 3: Implement minimal settings and wiring**

Add:

```python
groq_model: str = Field(
    default="qwen/qwen3.6-27b",
    validation_alias=AliasChoices("PY_WORKER_GROQ_MODEL", "GROQ_MODEL"),
)
groq_fallback_model: str | None = Field(
    default="llama-3.1-8b-instant",
    validation_alias=AliasChoices(
        "PY_WORKER_GROQ_FALLBACK_MODEL",
        "GROQ_FALLBACK_MODEL",
    ),
)
```

Pass `settings.groq_fallback_model` into `GroqClient`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_provider_routing -v`

Expected: all provider-routing tests pass.

### Task 2: Rate-Limit-Only Fallback

**Files:**
- Modify: `app/integrations/groq_client.py`
- Test: `tests/test_groq_client.py`

**Interfaces:**
- Consumes: `fallback_model: str | None` from dependency wiring.
- Produces: `GroqClient.generate_text(...) -> str`, with one fallback request after primary HTTP 429.

- [ ] **Step 1: Write failing HTTP behavior tests**

Add tests that record request model IDs and assert:

```python
self.assertEqual(models, ["qwen/qwen3.6-27b", "llama-3.1-8b-instant"])
```

for a primary 429 followed by success, and:

```python
self.assertEqual(models, ["qwen/qwen3.6-27b"])
```

for a primary 401.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_groq_client -v`

Expected: failure because `GroqClient` does not accept or use a fallback model.

- [ ] **Step 3: Implement the single retry**

Store `fallback_model`, extract request sending into a focused helper, and resend with the fallback only when `response.status_code == 429`, the fallback is non-empty, and it differs from the primary model.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_groq_client -v`

Expected: all Groq client tests pass.

### Task 3: Environment and Documentation

**Files:**
- Modify: `.env`
- Modify: `README.md`
- Modify: `../BACKEND/.env`

**Interfaces:**
- Produces: Docker-loaded environment values for both workers.

- [ ] **Step 1: Update explicit environment values**

Set:

```dotenv
GROQ_MODEL=qwen/qwen3.6-27b
GROQ_FALLBACK_MODEL=llama-3.1-8b-instant
GROQ_TOPIC_RESEARCH_MODEL=groq/compound-mini
```

- [ ] **Step 2: Update README configuration guidance**

Document the primary model, fallback model, and 429-only behavior.

- [ ] **Step 3: Verify environment parsing without exposing secrets**

Run focused `Select-String` commands that print only the three model settings.

### Task 4: Full Verification

**Files:**
- Verify all modified Python worker files and the backend environment setting.

**Interfaces:**
- Consumes: completed Tasks 1-3.
- Produces: fresh test and diff evidence.

- [ ] **Step 1: Run the complete Python worker test suite**

Run: `python -m unittest discover -s tests -v`

Expected: zero failures.

- [ ] **Step 2: Check formatting and unintended changes**

Run: `git diff --check`

Expected: exit code 0.

- [ ] **Step 3: Inspect the final focused diff**

Review only configuration, Groq client, tests, README, and model-only `.env` output before reporting completion.

