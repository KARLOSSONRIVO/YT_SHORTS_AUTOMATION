# Groq Script Duration Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Groq repair undersized narration using exact measured feedback while preserving the existing three-attempt duration gate.

**Architecture:** `LLMService` remains responsible for prompt construction, response parsing, and duration validation. It will derive a word budget and carry only the previous parsed narration’s measurement into the next prompt; the Groq integration and HTTP route remain unchanged.

**Tech Stack:** Python 3.11, Pydantic, `unittest`, FastAPI, Docker Compose

## Global Constraints

- Preserve `SCRIPT_MAX_ATTEMPTS = 3`.
- Preserve `MIN_DURATION_RATIO = 0.90` and `MAX_DURATION_RATIO = 1.08`.
- Preserve the configured Groq model and `max_new_tokens = 2200`.
- Do not relax validation or mechanically invent narration.
- Do not modify Gemini image generation or TTS.
- Preserve unrelated dirty working-tree changes and do not commit implementation files automatically.

---

### Task 1: Define duration-repair behavior

**Files:**
- Test: `tests/test_llm_service_duration.py`

**Interfaces:**
- Consumes: `LLMService.generate_story_script(payload)`.
- Produces: a regression test proving a short first response is repaired using measured feedback.

- [ ] Add a fake text client that returns a 54-word response unless its prompt contains `Previous narration word count: 54`, the literal acceptable range `112 to 133`, and the previous narration; then return a 120-word response.
- [ ] Call the real `LLMService` with a 60-second, 0.96-rate history payload and assert it returns narration inside the duration tolerance.
- [ ] Run `python -m unittest tests.test_llm_service_duration -v` and confirm it fails because the current retry prompt omits measured feedback.

### Task 2: Implement measured repair context

**Files:**
- Modify: `app/services/llm_service.py`
- Test: `tests/test_llm_service_duration.py`

**Interfaces:**
- Produces: an exact acceptable word range and feedback-rich retry prompt.
- Consumes: the last parsed `ScriptGenerationResponse` and its speaking-rate-aware duration estimate.

- [ ] Calculate minimum, target, and maximum word counts from duration, ratios, and speaking rate.
- [ ] Add the exact accepted range and a derived per-scene budget to every prompt.
- [ ] On retry, include previous word count, estimated duration, previous narration, and the correction needed to reach the center target rather than merely touching the closest validation boundary.
- [ ] Compare the top-level narration with combined scene narration and select the candidate closest to the target duration before validation and retry feedback.
- [ ] Run `python -m unittest tests.test_llm_service_duration -v` and confirm the regression test passes.

### Task 3: Verify and deploy

**Files:**
- Verify: `app/services/llm_service.py`
- Verify: `tests/test_llm_service_duration.py`

**Interfaces:**
- Produces: a rebuilt `python-worker` container.

- [ ] Run `python -m unittest discover -s tests -v`.
- [ ] Run `python -m compileall -q app tests` and `git diff --check`.
- [ ] Rebuild with `docker compose up -d --build python-worker`.
- [ ] Verify the live container has its Groq key and starts without errors.
- [ ] Send one 60-second Philippine-history request and print only HTTP status, narration word count, estimated duration, and scene count.
