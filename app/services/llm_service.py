import json
import re
from typing import Any

from app.core.exceptions import IntegrationError
from app.schemas.faceless_video import (
    FacelessScene,
    ScriptGenerationRequest,
    ScriptGenerationResponse,
)


class LLMService:
    SCRIPT_MAX_ATTEMPTS = 3
    MIN_DURATION_RATIO = 0.72
    ESTIMATED_WORDS_PER_SECOND = 2.15

    def __init__(
        self,
        *,
        llm_client: Any,
        model: str,
        allow_placeholder_generation: bool = False,
    ) -> None:
        self.llm_client = llm_client
        self.model = model
        self.allow_placeholder_generation = allow_placeholder_generation

    def generate_story_script(self, payload: ScriptGenerationRequest) -> ScriptGenerationResponse:
        try:
            last_response: ScriptGenerationResponse | None = None
            for attempt in range(1, self.SCRIPT_MAX_ATTEMPTS + 1):
                generated = self.llm_client.generate_text(
                    model=self.model,
                    prompt=self._build_prompt(payload, attempt=attempt),
                    max_new_tokens=2200,
                    temperature=0.75,
                )
                candidate = self._parse_response(payload, generated)
                if self._meets_duration_target(payload, candidate):
                    return candidate
                last_response = candidate

            if last_response is not None:
                estimated = round(self._estimate_narration_duration_seconds(last_response.narration), 1)
                minimum = round(payload.target_duration_seconds * self.MIN_DURATION_RATIO, 1)
                raise IntegrationError(
                    "Generated narration is too short for the requested duration. "
                    f"Estimated {estimated}s, expected at least {minimum}s."
                )

            raise IntegrationError("The LLM did not return a usable script.")
        except Exception as exc:
            if self.allow_placeholder_generation:
                return self._generate_placeholder_script(payload)
            if isinstance(exc, IntegrationError):
                raise
            raise IntegrationError(f"Hugging Face script generation failed: {exc}") from exc

    def detect_mood(self, script: str) -> str:
        normalized_script = re.sub(r"\s+", " ", script.lower())
        if any(token in normalized_script for token in ("ghost", "terror", "dark", "fear", "haunted", "murder")):
            return "horror"
        if any(token in normalized_script for token in ("grief", "loss", "heartbreak", "tragedy", "tears", "lonely")):
            return "sad"
        if any(token in normalized_script for token in ("epic", "legend", "battle", "empire", "dramatic", "cinematic")):
            return "cinematic"
        return "neutral"

    def _build_prompt(self, payload: ScriptGenerationRequest, *, attempt: int = 1) -> str:
        target_word_count = max(round(payload.target_duration_seconds * self.ESTIMATED_WORDS_PER_SECOND), 45)
        retry_instruction = ""
        if attempt > 1:
            retry_instruction = (
                "\nIMPORTANT RETRY INSTRUCTION:\n"
                f"- The previous attempt was too short for {payload.target_duration_seconds} seconds.\n"
                f"- Make the narration noticeably fuller and closer to {target_word_count} spoken words.\n"
                "- Add meaningful detail to every scene instead of shortening transitions.\n"
            )

        return f"""
You generate short-form faceless story videos for YouTube Shorts and TikTok.
Return only valid JSON. Do not wrap it in markdown.

Required JSON shape:
{{
  "title": "short title",
  "hook": "first sentence that grabs attention",
  "narration": "full narration script",
  "caption_text": "short social caption",
  "scenes": [
    {{
      "scene_index": 1,
      "narration": "scene narration",
      "image_prompt": "vertical 9:16 cinematic visual prompt, no text, no logos",
      "duration_seconds": 6,
      "caption_text": "short subtitle text"
    }}
  ]
}}

Topic: {payload.topic}
Tone: {payload.tone}
Language: {payload.language}
Target duration seconds: {payload.target_duration_seconds}
Approximate target spoken word count: {target_word_count}
Visual style preset: {payload.style_preset}
Audience: {payload.audience or "general short-form viewers"}

Rules:
- Create 3 to 8 scenes.
- Keep every scene narration concise and spoken aloud naturally.
- Make the full narration long enough to fill roughly {payload.target_duration_seconds} seconds of voice-over.
- Every image prompt must describe one specific frozen visual moment with:
  - the main subject
  - the exact action happening
  - the setting/background
  - camera framing or angle
  - lighting / mood
- Image prompts must be concrete and physically believable, not abstract or symbolic.
- If the topic is about sports, the prompt must name the correct sport, equipment, field/court, and realistic action pose.
- Avoid extra limbs, broken anatomy, floating objects, duplicated people, visible text, logos, watermarks, scoreboards, UI overlays, or subtitles.
- The total scene durations should be close to the target duration.
- Return JSON only.
{retry_instruction}
""".strip()

    def _parse_response(
        self,
        payload: ScriptGenerationRequest,
        generated_text: str,
    ) -> ScriptGenerationResponse:
        data = self._extract_json(generated_text)
        scenes = [
            FacelessScene(
                scene_index=int(scene.get("scene_index", index + 1)),
                narration=str(scene.get("narration", "")).strip(),
                image_prompt=str(scene.get("image_prompt", "")).strip(),
                duration_seconds=float(scene.get("duration_seconds", 6.0)),
                caption_text=str(scene.get("caption_text", scene.get("narration", ""))).strip(),
            )
            for index, scene in enumerate(data.get("scenes", []))
            if isinstance(scene, dict)
        ]
        scenes = [scene for scene in scenes if scene.narration and scene.image_prompt]

        if not scenes:
            raise IntegrationError("The LLM did not return any valid scenes.")

        narration = str(data.get("narration") or " ".join(scene.narration for scene in scenes)).strip()
        title = str(data.get("title") or payload.topic).strip()
        hook = str(data.get("hook") or scenes[0].narration).strip()
        caption_text = str(data.get("caption_text") or hook).strip()

        return ScriptGenerationResponse(
            job_id=payload.job_id,
            project_id=payload.project_id,
            title=title,
            hook=hook,
            narration=narration,
            scenes=scenes,
            image_prompts=[scene.image_prompt for scene in scenes],
            caption_text=caption_text,
        )

    def _estimate_narration_duration_seconds(self, narration: str) -> float:
        words = re.findall(r"\b[\w']+\b", narration)
        if not words:
            return 0.0
        return len(words) / self.ESTIMATED_WORDS_PER_SECOND

    def _meets_duration_target(
        self,
        payload: ScriptGenerationRequest,
        response: ScriptGenerationResponse,
    ) -> bool:
        estimated_duration = self._estimate_narration_duration_seconds(response.narration)
        minimum_duration = payload.target_duration_seconds * self.MIN_DURATION_RATIO
        return estimated_duration >= minimum_duration

    def _extract_json(self, generated_text: str) -> dict:
        # Strip <think>...</think> blocks emitted by reasoning models (e.g. Qwen3.5, DeepSeek-R1)
        cleaned = re.sub(r"<think>.*?</think>", "", generated_text, flags=re.DOTALL).strip()

        # Strip markdown code fences (```json ... ```)
        if "```" in cleaned:
            cleaned = re.sub(r"```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r"```", "", cleaned).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Last resort: grab the first {...} block in the text
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                raise IntegrationError(
                    f"The LLM response did not contain a JSON object. "
                    f"Raw response (first 500 chars): {generated_text[:500]}"
                )
            data = json.loads(match.group(0))

        if not isinstance(data, dict):
            raise IntegrationError("The LLM response JSON must be an object.")
        return data

    def _generate_placeholder_script(self, payload: ScriptGenerationRequest) -> ScriptGenerationResponse:
        topic = payload.topic.strip()
        audience = f" for {payload.audience.strip()}" if payload.audience else ""
        scene_count = min(max(round(payload.target_duration_seconds / 7), 3), 8)
        scene_duration = max(payload.target_duration_seconds / scene_count, 3)

        title = f"The hidden story of {topic.title()}"
        hook = f"What if everything you knew about {topic} was only the surface?"
        scenes: list[FacelessScene] = []

        beats = [
            "Open with the mystery and a detail that feels impossible to ignore.",
            "Reveal the ordinary world before the first strange turn.",
            "Introduce the pressure, the cost, and the choice nobody wanted.",
            "Escalate with a surprising consequence that changes the stakes.",
            "Slow down for the emotional truth behind the story.",
            "Deliver the twist, insight, or lesson that makes the viewer stay.",
            "Close with a memorable image and a question worth sharing.",
            "Leave the audience with a clean final thought.",
        ]

        for index in range(scene_count):
            beat = beats[index % len(beats)]
            narration = (
                f"{beat} This moment in {topic} matters because it turns a simple idea "
                f"into something people remember{audience}."
            )
            scenes.append(
                FacelessScene(
                    scene_index=index + 1,
                    narration=narration,
                    image_prompt=(
                        f"{payload.style_preset}, vertical 9:16 scene about {topic}, "
                        f"{beat.lower()}, moody lighting, no text, no logos"
                    ),
                    duration_seconds=round(scene_duration, 2),
                    caption_text=narration[:140],
                )
            )

        narration = " ".join([hook, *[scene.narration for scene in scenes]])
        return ScriptGenerationResponse(
            job_id=payload.job_id,
            project_id=payload.project_id,
            title=title,
            hook=hook,
            narration=narration,
            scenes=scenes,
            image_prompts=[scene.image_prompt for scene in scenes],
            caption_text=f"{hook} #{topic.replace(' ', '')[:32]}",
        )
