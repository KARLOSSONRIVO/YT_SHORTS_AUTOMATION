import json
import math
import re
from typing import Any

from app.core.exceptions import IntegrationError
from app.schemas.faceless_video import (
    FacelessScene,
    ScriptGenerationRequest,
    ScriptGenerationResponse,
)


class LLMService:
    SCRIPT_MAX_ATTEMPTS = 2
    SCRIPT_MAX_NEW_TOKENS = 1600
    MIN_DURATION_RATIO = 0.90
    MAX_DURATION_RATIO = 1.08
    ESTIMATED_WORDS_PER_SECOND = 2.15
    MAX_PSYCHOLOGY_HOOK_WORDS = 8
    MAX_HISTORY_HOOK_WORDS = 9

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
            best_response: ScriptGenerationResponse | None = None
            best_duration_error = float("inf")
            adjustment = "expand"
            for attempt in range(1, self.SCRIPT_MAX_ATTEMPTS + 1):
                try:
                    generated = self.llm_client.generate_text(
                        model=self.model,
                        prompt=self._build_prompt(
                            payload,
                            attempt=attempt,
                            adjustment=adjustment,
                            previous_response=last_response,
                        ),
                        max_new_tokens=self.SCRIPT_MAX_NEW_TOKENS,
                        temperature=0.75,
                    )
                    candidate = self._parse_response(payload, generated)
                except IntegrationError:
                    if best_response is not None:
                        return best_response
                    raise
                estimated_duration = self._estimate_narration_duration_seconds(
                    candidate.narration,
                    payload.speaking_rate,
                )
                duration_error = abs(
                    estimated_duration - payload.target_duration_seconds
                )
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

            raise IntegrationError("The LLM did not return a usable script.")
        except Exception as exc:
            if self.allow_placeholder_generation:
                return self._generate_placeholder_script(payload)
            if isinstance(exc, IntegrationError):
                raise
            raise IntegrationError(f"Story script generation failed: {exc}") from exc

    def detect_mood(self, script: str) -> str:
        normalized_script = re.sub(r"\s+", " ", script.lower())
        if any(token in normalized_script for token in ("ghost", "terror", "dark", "fear", "haunted", "murder")):
            return "horror"
        if any(token in normalized_script for token in ("grief", "loss", "heartbreak", "tragedy", "tears", "lonely")):
            return "sad"
        if any(token in normalized_script for token in ("epic", "legend", "battle", "empire", "dramatic", "cinematic")):
            return "cinematic"
        return "neutral"

    def _build_prompt(
        self,
        payload: ScriptGenerationRequest,
        *,
        attempt: int = 1,
        adjustment: str = "expand",
        previous_response: ScriptGenerationResponse | None = None,
    ) -> str:
        target_word_count = max(round(payload.target_duration_seconds * self.ESTIMATED_WORDS_PER_SECOND * payload.speaking_rate), 45)
        minimum_word_count = math.ceil(
            payload.target_duration_seconds
            * self.MIN_DURATION_RATIO
            * self.ESTIMATED_WORDS_PER_SECOND
            * payload.speaking_rate
        )
        maximum_word_count = max(
            math.floor(
                payload.target_duration_seconds
                * self.MAX_DURATION_RATIO
                * self.ESTIMATED_WORDS_PER_SECOND
                * payload.speaking_rate
            ),
            minimum_word_count,
        )
        repair_target_word_count = min(
            max(target_word_count, minimum_word_count),
            maximum_word_count,
        )
        minimum_scene_words = max(math.ceil(minimum_word_count / 8), 1)
        maximum_scene_words = max(math.ceil(maximum_word_count / 6), minimum_scene_words)
        duration_instruction = (
            f"Target narration word count: {repair_target_word_count}.\n"
            f"Acceptable full narration range: {minimum_word_count} to {maximum_word_count} spoken words.\n"
            f"For 6 to 8 scenes, average {minimum_scene_words} to {maximum_scene_words} spoken words per scene. "
            "The combined scene narration must cover the full narration without omitting spoken lines."
        )
        retry_instruction = ""
        if attempt > 1 and previous_response is not None:
            previous_word_count = self._narration_word_count(previous_response.narration)
            previous_duration = self._estimate_narration_duration_seconds(
                previous_response.narration,
                payload.speaking_rate,
            )
            if previous_word_count < minimum_word_count:
                correction = (
                    f"Add at least {repair_target_word_count - previous_word_count} meaningful spoken words "
                    "while preserving factual accuracy."
                )
            elif previous_word_count > maximum_word_count:
                correction = (
                    f"Remove at least {previous_word_count - repair_target_word_count} lower-value spoken words "
                    "without losing the story's key facts."
                )
            else:
                correction = "Rewrite the narration so its pacing fits the requested duration."
            retry_instruction = f"""
IMPORTANT RETRY:
Previous narration word count: {previous_word_count}
Previous estimated duration: {previous_duration:.1f} seconds
Target narration word count: {repair_target_word_count}
Acceptable full narration range: {minimum_word_count} to {maximum_word_count} spoken words
Required correction: {correction}
Previous narration to revise:
---BEGIN PREVIOUS NARRATION---
{previous_response.narration}
---END PREVIOUS NARRATION---
Return a complete revised JSON response, not commentary about the revision.
"""
        elif attempt > 1:
            action = "Remove repetition and compress lower-value detail" if adjustment == "shorten" else "Add meaningful factual detail to every scene"
            retry_instruction = f"\nIMPORTANT RETRY: The prior narration was outside {payload.target_duration_seconds}s. {action}; target {target_word_count} spoken words.\n"

        if payload.script_framework == "reddit_story":
            return self._build_reddit_story_prompt(
                payload,
                target_word_count=target_word_count,
                duration_instruction=duration_instruction,
                retry_instruction=retry_instruction,
            )

        if payload.script_framework == "history_story":
            return self._build_history_story_prompt(
                payload,
                target_word_count=target_word_count,
                duration_instruction=duration_instruction,
                retry_instruction=retry_instruction,
            )

        return self._build_psychology_truth_prompt(
            payload,
            target_word_count=target_word_count,
            duration_instruction=duration_instruction,
            retry_instruction=retry_instruction,
        )

    def _build_standard_story_prompt(
        self,
        payload: ScriptGenerationRequest,
        *,
        target_word_count: int,
        duration_instruction: str,
        retry_instruction: str,
    ) -> str:
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
Selected story format: {payload.story_format or "psychological_explanation"}
Tone: {payload.tone}
Language: {payload.language}
Target duration seconds: {payload.target_duration_seconds}
Approximate target spoken word count: {target_word_count}
{duration_instruction}
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

    def _build_psychology_truth_prompt(
        self,
        payload: ScriptGenerationRequest,
        *,
        target_word_count: int,
        duration_instruction: str,
        retry_instruction: str,
    ) -> str:
        return f"""
You generate retention-first faceless psychology shorts for YouTube Shorts and TikTok.
Return only valid JSON. Do not wrap it in markdown.

Required JSON shape:
{{
  "title": "short curiosity title for the project metadata",
  "hook": "short punchy opening hook, 4 to 8 words maximum",
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
Selected story format: {payload.story_format or "psychological_explanation"}
Tone: {payload.tone}
Language: {payload.language}
Target duration seconds: {payload.target_duration_seconds}
Approximate target spoken word count: {target_word_count}
{duration_instruction}
Visual style preset: {payload.style_preset}
Audience: {payload.audience or "viewers who respond to blunt psychology truths"}

Beat formula to follow:
1. Opening hook: use a short punchy line that creates curiosity without explaining the whole topic.
2. Set up the conflict: explain the psychological pattern and why it matters personally.
3. Mid-video hook #1: pattern interrupt or sharper reframing.
4. Truth bomb: one uncomfortable but believable psychology fact.
5. Mid-video hook #2: curiosity spike that raises the stakes.
6. Breakdown: explain why the brain or mind behaves this way in simple emotional language.
7. Wake-up slap: one line that feels confronting but useful.
8. Solution: one specific mindset shift or action.
9. Closing hook: a final emotional line that lingers, not a generic call to action.

Rules:
- Start directly with the spoken hook. Do NOT speak the title in the narration.
- Keep the hook simple and direct. Do not summarize the full topic in the hook.
- Do not repeat the title, project name, or full topic inside the hook.
- Good hook examples: "Your brain is lying again.", "This is why it hurts.", "Most people miss this.", "This pattern controls you."
- Create 6 to 8 scenes.
- Each scene should map naturally to one of the beats above.
- Make the script sound like a smooth spoken short, not labeled sections.
- The hook must land in the very first sentence.
- The truth bomb should be emotionally sharp but not melodramatic.
- The solution must be practical and specific.
- The closing line should feel like a final mental punch.
- Make the full narration long enough to fill roughly {payload.target_duration_seconds} seconds of voice-over.
- Every image prompt must describe one specific frozen visual moment with:
  - the main subject
  - the exact action happening
  - the setting/background
  - camera framing or angle
  - lighting / mood
- For psychology scenes, prefer realistic human behavior, body language, tension, reflection, confrontation, isolation, social pressure, rejection, or breakthrough moments.
- The image prompt must explicitly avoid any text-bearing elements in the frame: no signs, no captions, no subtitles, no posters, no screens with text, no scoreboard text, no jersey names, no jersey numbers, no uniform lettering, no labels, no banners.
- Avoid extra limbs, broken anatomy, floating objects, duplicated people, visible text, logos, watermarks, scoreboards, UI overlays, or subtitles.
- The total scene durations should be close to the target duration.
- Return JSON only.
{retry_instruction}
""".strip()

    def _build_reddit_story_prompt(
        self,
        payload: ScriptGenerationRequest,
        *,
        target_word_count: int,
        duration_instruction: str,
        retry_instruction: str,
    ) -> str:
        source_title = payload.project_title or payload.topic
        source_text = payload.source_text or payload.topic
        return f"""
You create a short-form dramatized retelling of a Reddit submission for YouTube Shorts and TikTok.
Return only valid JSON. Do not wrap it in markdown.

Required JSON shape:
{{
  "title": "a concise title that stays close to the Reddit submission title",
  "hook": "short punchy opening hook",
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

Reddit submission title: {source_title}
Reddit submission text: {source_text}
Selected story format: {payload.story_format or "unexpected_ending"}
Tone: {payload.tone}
Language: {payload.language}
Target duration seconds: {payload.target_duration_seconds}
Approximate target spoken word count: {target_word_count}
{duration_instruction}
Visual style preset: {payload.style_preset}
Audience: {payload.audience or "viewers who enjoy emotionally engaging Reddit stories"}

Rules:
- Use the Reddit submission above as the only story source.
- Do not turn this into history, war, politics, mythology, or a different unrelated story.
- Do not invent historical names, dates, battles, countries, uniforms, or events.
- Preserve the submission's central people, relationships, situation, and decision.
- If the feed contains only a title, create a clearly dramatized fictional retelling of that title while keeping its premise unchanged.
- Keep the title close to the Reddit submission title; never replace it with an unrelated title.
- Do not claim the anonymous submission is independently verified.
- Start directly with the spoken hook and create 6 to 8 scenes.
- Make every image prompt describe one specific frozen moment with realistic human behavior and no visible text, logos, watermarks, or subtitles.
- Return JSON only.
{retry_instruction}
""".strip()

    def _build_history_story_prompt(
        self,
        payload: ScriptGenerationRequest,
        *,
        target_word_count: int,
        duration_instruction: str,
        retry_instruction: str,
    ) -> str:
        return f"""
You generate retention-first faceless history shorts for YouTube Shorts and TikTok.
Return only valid JSON. Do not wrap it in markdown.

Required JSON shape:
{{
  "title": "short curiosity title for the project metadata",
  "hook": "short punchy opening hook, 4 to 9 words maximum",
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
Selected story format: {payload.story_format or "hidden_history"}
Tone: {payload.tone}
Language: {payload.language}
Target duration seconds: {payload.target_duration_seconds}
Approximate target spoken word count: {target_word_count}
{duration_instruction}
Visual style preset: {payload.style_preset}
Audience: {payload.audience or "viewers who enjoy dramatic history stories"}

Beat formula to follow:
Adapt the beats to the selected story format. Examples: record_breaking_moment = Hook, Context, Challenge, Record, Outcome, Twist; mystery/unsolved = Hook, Setting, Strange Event, Evidence, Main Theory, Unresolved Ending; rise_and_fall = Peak, Origin, Escalation, Fatal Choice, Collapse, Legacy.
1. Opening hook: a short line that sparks immediate curiosity.
2. Historical setup: who, where, and what moment we are entering.
3. Rising tension: what was going wrong or what danger was building.
4. Turning point: the decision, mistake, betrayal, battle, or event that changed everything.
5. Consequence: what happened right after and why it shocked people.
6. Legacy: why the story still matters, what it changed, or why history remembers it.
7. Closing sting: one final line that leaves the viewer with the weight of the event.

Rules:
- Start directly with the spoken hook. Do NOT speak the title in the narration.
- Keep the hook simple and direct. Do not summarize the full topic in the hook.
- Do not repeat the title, project name, or full topic inside the hook.
- Good hook examples: "History almost forgot this.", "One mistake changed everything.", "This should never have happened.", "An empire cracked here."
- Create 6 to 8 scenes.
- Each scene should map naturally to one of the beats above.
- Make the script sound like a smooth spoken short, not labeled sections.
- The hook must land in the very first sentence.
- Make the history vivid, concrete, and easy to follow without sounding like a textbook.
- Focus on real people, pressure, consequences, and stakes.
- The closing line should make the event feel meaningful or haunting.
- Make the full narration long enough to fill roughly {payload.target_duration_seconds} seconds of voice-over.
- Every image prompt must describe one specific frozen visual moment with:
  - the main subject
  - the exact action happening
  - the setting/background
  - camera framing or angle
  - lighting / mood
- For history scenes, prefer believable period details, clothing, architecture, tools, battlefields, courts, ships, crowds, maps, smoke, torchlight, ruins, and human expressions.
- The image prompt must explicitly avoid any text-bearing elements in the frame: no signs, no captions, no subtitles, no posters, no screens with text, no scoreboard text, no jersey names, no jersey numbers, no uniform lettering, no labels, no banners.
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

        top_level_narration = str(data.get("narration") or "").strip()
        title = str(data.get("title") or payload.topic).strip()
        hook = str(data.get("hook") or scenes[0].narration).strip()
        caption_text = str(data.get("caption_text") or hook).strip()

        if payload.script_framework == "history_story":
            hook = self._normalize_history_hook(payload.topic, hook)
            title = self._normalize_history_title(payload.topic, title)
            scenes = self._ensure_psychology_scene_lead(hook, scenes)
        else:
            hook = self._normalize_psychology_hook(payload.topic, hook)
            title = self._normalize_psychology_title(payload.topic, title)
            scenes = self._ensure_psychology_scene_lead(hook, scenes)

        scene_narration = " ".join(scene.narration for scene in scenes).strip()
        normalized_top_level_narration = self._ensure_psychology_hook_lead(
            hook,
            top_level_narration or scene_narration,
        )
        narration = self._select_narration_closest_to_target(
            payload,
            normalized_top_level_narration,
            scene_narration,
        )

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

    def _estimate_narration_duration_seconds(self, narration: str, speaking_rate: float = 1.0) -> float:
        word_count = self._narration_word_count(narration)
        if not word_count:
            return 0.0
        return word_count / (self.ESTIMATED_WORDS_PER_SECOND * speaking_rate)

    def _narration_word_count(self, narration: str) -> int:
        return len(re.findall(r"\b[\w']+\b", narration))

    def _select_narration_closest_to_target(
        self,
        payload: ScriptGenerationRequest,
        *candidates: str,
    ) -> str:
        usable_candidates = [candidate.strip() for candidate in candidates if candidate.strip()]
        if not usable_candidates:
            return ""
        return min(
            usable_candidates,
            key=lambda candidate: abs(
                self._estimate_narration_duration_seconds(candidate, payload.speaking_rate)
                - payload.target_duration_seconds
            ),
        )

    def _meets_duration_target(
        self,
        payload: ScriptGenerationRequest,
        response: ScriptGenerationResponse,
    ) -> bool:
        estimated_duration = self._estimate_narration_duration_seconds(response.narration, payload.speaking_rate)
        minimum_duration = payload.target_duration_seconds * self.MIN_DURATION_RATIO
        maximum_duration = payload.target_duration_seconds * self.MAX_DURATION_RATIO
        return minimum_duration <= estimated_duration <= maximum_duration

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
        if payload.script_framework == "history_story":
            return self._generate_placeholder_history_script(payload)
        return self._generate_placeholder_psychology_script(payload)

    def _generate_placeholder_history_script(self, payload: ScriptGenerationRequest) -> ScriptGenerationResponse:
        topic = payload.topic.strip()
        tone = (payload.tone or "dramatic, vivid, cinematic").strip()
        scene_count = min(max(round(payload.target_duration_seconds / 7), 6), 8)
        scene_duration = max(payload.target_duration_seconds / scene_count, 3)

        title = topic or "The history they almost forgot"
        hook = self._fallback_history_hook(topic)
        beats = [
            hook,
            f"It started with {topic}, in a moment people thought they understood, but the real danger was only beginning.",
            "The pressure built quietly, and the people inside the story had less time, less control, and fewer good choices than anyone realized.",
            "Then came the turning point, the one decision that changed the course of everything that followed.",
            "What happened next stunned everyone watching, because the cost was bigger and faster than anyone expected.",
            "The fallout did not end in that moment. It kept spreading through the people, the place, and the future it touched.",
            "That is why history kept this story alive, not because it was ordinary, but because it exposed how fragile power and certainty really are.",
            "And once you see how it happened, it becomes impossible to pretend it could never happen again.",
        ]
        visuals = [
            "single central figure in a tense historical moment, cinematic close-up, dramatic light",
            "wide period environment establishing where the event unfolds, believable architecture and clothing",
            "crowd tension, worried faces, rising pressure, grounded historical atmosphere",
            "the decisive action or mistake frozen at the critical instant, dynamic framing",
            "immediate aftermath, shock, smoke, chaos, or stunned silence, emotional realism",
            "aftermath spreading through a city, battlefield, court, harbor, or public square",
            "symbolic but physically believable reminder of the legacy, ruins, memorial, map, or surviving witness",
            "quiet closing image that leaves emotional weight, lone figure, fading torchlight, or haunted landscape",
        ]

        scenes: list[FacelessScene] = []
        for index in range(scene_count):
            narration = beats[index]
            scenes.append(
                FacelessScene(
                    scene_index=index + 1,
                    narration=narration,
                    image_prompt=(
                        f"{payload.style_preset}, {tone}, vertical 9:16, {visuals[index]}, "
                        "historically grounded, realistic human proportions, no text, no logos"
                    ),
                    duration_seconds=round(scene_duration, 2),
                    caption_text=narration[:140],
                )
            )

        narration = " ".join(scene.narration for scene in scenes)
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

    def _generate_placeholder_psychology_script(self, payload: ScriptGenerationRequest) -> ScriptGenerationResponse:
        topic = payload.topic.strip()
        tone = (payload.tone or "direct, emotionally honest").strip()
        scene_count = min(max(round(payload.target_duration_seconds / 7), 6), 8)
        scene_duration = max(payload.target_duration_seconds / scene_count, 3)

        title = f"The psychology behind {topic.lower()}"
        hook = f"If you keep struggling with {topic}, this harsh psychology truth is going to hurt."
        beats = [
            hook,
            f"Psychologists would tell you this pattern around {topic} isn't random. It quietly shapes your confidence, your habits, and your relationships.",
            "But here's where it gets worse: your brain confuses what feels familiar with what is actually good for you.",
            "That means you can stay loyal to the very pattern that keeps disappointing you, just because discomfort feels threatening.",
            "And this next part explains why change feels so hard: your mind would rather protect you from uncertainty than help you grow.",
            "That is why you hesitate, overthink, and repeat behavior you already know is costing you.",
            "If that hit you, good. It means you finally saw the pattern instead of defending it.",
            f"If you want to break it, choose one small uncomfortable action around {topic} before your brain has time to negotiate you back into the old cycle.",
        ]
        visuals = [
            "person frozen mid-thought, intense expression, quiet room, cinematic close-up",
            "person sitting across from another in a tense conversation, subtle body language, realistic social pressure",
            "late-night isolation, person staring at phone in dark room, emotional discomfort, over-the-shoulder framing",
            "person choosing the familiar path despite visible hesitation, grounded realistic environment, dramatic light",
            "split-second hesitation before speaking up in a meeting, shallow depth of field, emotional tension",
            "reflection in a mirror, tired eyes, intimate framing, visible internal conflict",
            "moment of realization, person sitting upright with determined expression, moody but hopeful light",
            "person taking a difficult first step forward, realistic body language, subtle breakthrough energy",
        ]

        scenes: list[FacelessScene] = []
        for index in range(scene_count):
            narration = beats[index]
            scenes.append(
                FacelessScene(
                    scene_index=index + 1,
                    narration=narration,
                    image_prompt=(
                        f"{payload.style_preset}, {tone}, vertical 9:16, {visuals[index]}, "
                        "realistic human proportions, no text, no logos"
                    ),
                    duration_seconds=round(scene_duration, 2),
                    caption_text=narration[:140],
                )
            )

        narration = " ".join(scene.narration for scene in scenes)
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

    def _normalize_psychology_hook(self, topic: str, hook: str) -> str:
        normalized_hook = re.sub(r"\s+", " ", hook).strip()
        if normalized_hook:
            first_sentence = re.split(r"(?<=[.!?])\s+", normalized_hook)[0].strip()
            shortened = self._shorten_psychology_hook(first_sentence)
            if shortened:
                return shortened

        return self._fallback_psychology_hook(topic)

    def _shorten_psychology_hook(self, hook: str) -> str:
        cleaned = re.sub(r"\s+", " ", hook).strip()
        if not cleaned:
            return ""

        words = cleaned.split()
        if len(words) <= self.MAX_PSYCHOLOGY_HOOK_WORDS:
            return cleaned

        lowered = cleaned.lower()
        if "ignore" in lowered or "lose interest" in lowered or "pull away" in lowered:
            return "This is why it hurts."
        if "confidence" in lowered:
            return "Your confidence is being hijacked."
        if "habit" in lowered or "lazy" in lowered:
            return "Your brain is protecting the habit."
        if "relationship" in lowered:
            return "This pattern controls you."

        return "Your brain is lying again."

    def _fallback_psychology_hook(self, topic: str) -> str:
        lowered = topic.lower()
        if any(token in lowered for token in ("ignore", "lose interest", "pull away", "rejection")):
            return "This is why it hurts."
        if "confidence" in lowered:
            return "Your confidence is being hijacked."
        if any(token in lowered for token in ("habit", "lazy", "procrastination")):
            return "Your brain is protecting the habit."
        if "relationship" in lowered:
            return "This pattern controls you."

        return "Your brain is lying again."

    def _normalize_history_hook(self, topic: str, hook: str) -> str:
        normalized_hook = re.sub(r"\s+", " ", hook).strip()
        if normalized_hook:
            first_sentence = re.split(r"(?<=[.!?])\s+", normalized_hook)[0].strip()
            shortened = self._shorten_history_hook(first_sentence)
            if shortened:
                return shortened

        return self._fallback_history_hook(topic)

    def _shorten_history_hook(self, hook: str) -> str:
        cleaned = re.sub(r"\s+", " ", hook).strip()
        if not cleaned:
            return ""

        words = cleaned.split()
        if len(words) <= self.MAX_HISTORY_HOOK_WORDS:
            return cleaned

        lowered = cleaned.lower()
        if any(token in lowered for token in ("empire", "kingdom", "dynasty", "rome", "throne")):
            return "An empire cracked here."
        if any(token in lowered for token in ("war", "battle", "siege", "army")):
            return "One battle changed everything."
        if any(token in lowered for token in ("betray", "traitor", "plot", "assassin")):
            return "One betrayal changed history."
        if any(token in lowered for token in ("ship", "ocean", "voyage", "expedition")):
            return "This voyage went terribly wrong."

        return "History almost forgot this."

    def _fallback_history_hook(self, topic: str) -> str:
        lowered = topic.lower()
        if any(token in lowered for token in ("empire", "kingdom", "dynasty", "rome", "throne")):
            return "An empire cracked here."
        if any(token in lowered for token in ("war", "battle", "siege", "army")):
            return "One battle changed everything."
        if any(token in lowered for token in ("betray", "traitor", "plot", "assassin")):
            return "One betrayal changed history."
        if any(token in lowered for token in ("ship", "ocean", "voyage", "expedition")):
            return "This voyage went terribly wrong."

        return "History almost forgot this."

    def _normalize_history_title(self, topic: str, title: str) -> str:
        normalized = title.strip()
        if normalized:
            return normalized
        return topic.strip() or "The history they almost forgot"

    def _normalize_psychology_title(self, topic: str, title: str) -> str:
        normalized = title.strip()
        if normalized:
            return normalized
        return f"The psychology behind {topic}".strip()

    def _ensure_psychology_hook_lead(self, hook: str, narration: str) -> str:
        cleaned_narration = narration.strip()
        if not cleaned_narration:
            return hook

        if self._normalize_text(cleaned_narration[: max(len(hook) + 40, 160)]).startswith(
            self._normalize_text(hook)
        ):
            return cleaned_narration

        return f"{hook} {cleaned_narration}".strip()

    def _ensure_psychology_scene_lead(
        self,
        hook: str,
        scenes: list[FacelessScene],
    ) -> list[FacelessScene]:
        if not scenes:
            return scenes

        first_scene = scenes[0]
        if self._normalize_text(first_scene.narration).startswith(self._normalize_text(hook)):
            return scenes

        updated_first_scene = FacelessScene(
            scene_index=first_scene.scene_index,
            narration=hook,
            image_prompt=first_scene.image_prompt,
            duration_seconds=first_scene.duration_seconds,
            caption_text=hook,
        )
        return [updated_first_scene, *scenes[1:]]

    def _normalize_text(self, value: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s']+", " ", value.lower())).strip()
