# Groq Script Model Fallback Design

## Goal

Use `qwen/qwen3.6-27b` as the Python worker's primary script-generation model and retry the same generation request once with `llama-3.1-8b-instant` only when the primary request is rate-limited. Keep backend topic research on `groq/compound-mini`.

## Configuration

- `GROQ_MODEL` / `PY_WORKER_GROQ_MODEL`: primary script model, default `qwen/qwen3.6-27b`.
- `GROQ_FALLBACK_MODEL` / `PY_WORKER_GROQ_FALLBACK_MODEL`: rate-limit fallback, default `llama-3.1-8b-instant`.
- `GROQ_TOPIC_RESEARCH_MODEL`: backend research model, explicitly set to `groq/compound-mini`.

The local `.env` files will explicitly contain these values so recreated containers receive them.

## Request Flow

`LLMService` continues to make one logical script-generation call through `GroqClient`. The client sends the request to the requested primary model. If and only if Groq responds with HTTP 429 and the configured fallback differs from the primary model, the client repeats the same prompt and generation settings once using the fallback model.

All non-429 errors remain failures. A failed fallback also remains a failure; there is no fallback loop and no Gemini text-generation fallback.

## Testing

Automated tests will verify:

- Qwen is the default primary model and Llama 3.1 8B is the default fallback.
- dependency wiring passes the fallback model into `GroqClient`.
- a 429 response sends exactly one second request using the fallback model.
- non-429 errors do not trigger fallback.
- the existing JSON request contract remains intact.

