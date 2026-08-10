# Groq Script Duration Repair Design

## Goal

Generate narration that reliably satisfies the existing 90%–108% duration tolerance instead of failing after three valid but undersized Groq responses.

## Evidence

The Python worker received the Groq key and Groq returned HTTP 200 for all three attempts. The final parsed history script was estimated at 26.2 seconds for a 60-second target, so `LLMService` correctly rejected it. The current retries say only that the prior result was outside tolerance and regenerate from scratch without the measured word count or previous narration.

## Selected Approach

- Compute an acceptable spoken-word range from the requested duration, speaking rate, and existing duration ratios. For 60 seconds at 0.96 speaking rate, the range is 112–133 words.
- Put the exact range in every generation prompt and require scene narration to collectively cover the full narration.
- When an attempt misses the range, provide the next attempt with the previous word count, estimated duration, exact deficit or excess, and previous narration to revise.
- Keep the existing three-attempt limit, validation ratios, Groq model, and 2,200-token response ceiling.

This directly addresses the observed failure while avoiding extra provider calls.

## Rejected Approaches

- Relaxing the duration validator would allow visibly short videos and does not make the generated narration meet the user’s 60-second setting.
- Blindly increasing the attempt count adds cost and repeats the same under-specified request.
- Padding narration mechanically would risk repetition and unsupported historical claims.

## Data Flow

1. `LLMService` calculates the target and acceptable word range.
2. Groq receives the range plus a per-scene spoken-word budget.
3. The parsed response is measured with the existing speaking-rate-aware estimator.
4. A response inside tolerance returns normally.
5. A short or long response becomes the repair context for the next attempt.
6. After three misses, the existing `IntegrationError` remains visible.

## Testing and Deployment

A feedback-aware fake client returns a 26-second response unless the retry prompt contains the measured shortfall, accepted range, and prior narration. The service must recover on the second attempt and return a valid-duration response. Run all Python unit tests, compile the changed files, rebuild only `python-worker`, and issue one live 60-second Philippine-history script request that prints duration and word-count summaries only.

## Scope

Backend Groq topic research, Gemini image generation, Gemini TTS, Redis data, and the duration tolerance itself are unchanged.
