from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any


@dataclass(slots=True)
class AudioSceneMetadata:
    mood: str
    environment: str
    emotional_tone: str
    tension_level: float
    ambience_prompt: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AudioSceneAnalysisService:
    MOOD_KEYWORDS = {
        "horror": ("haunted", "ghost", "terror", "eerie", "nightmare", "blood", "shadow", "stalk"),
        "sad": ("lonely", "grief", "loss", "cry", "tears", "heartbreak", "abandoned"),
        "cinematic": ("battle", "epic", "legend", "empire", "dramatic", "hero", "fate"),
        "fantasy": ("magic", "wizard", "dragon", "spell", "enchanted", "kingdom", "portal"),
        "sci-fi": ("spaceship", "space", "alien", "robot", "cyber", "neon", "station", "future"),
        "calm": ("peaceful", "gentle", "quiet", "soft", "warm", "sunset", "serene"),
    }
    ENVIRONMENT_KEYWORDS = {
        "forest": ("forest", "woods", "jungle", "trees", "crows", "leaves"),
        "city": ("city", "street", "alley", "traffic", "neon", "rooftop", "downtown"),
        "rainy city": ("rain", "rainy", "storm", "wet street", "thunder"),
        "spaceship": ("spaceship", "spacecraft", "station", "airlock", "engine room"),
        "cave": ("cave", "cavern", "underground", "tunnel", "crystal"),
        "ocean": ("ocean", "sea", "waves", "beach", "shore", "harbor"),
        "room": ("room", "house", "apartment", "bedroom", "hallway", "office"),
        "desert": ("desert", "sand", "dune", "wasteland"),
    }
    TENSION_KEYWORDS = {
        "low": ("peaceful", "calm", "gentle", "quiet", "warm"),
        "medium": ("mysterious", "uncertain", "dramatic", "strange", "secret"),
        "high": ("terror", "chase", "panic", "attack", "scream", "danger", "haunted", "monster"),
    }
    ENVIRONMENT_TEXTURES = {
        "forest": "wind through trees, distant birds, soft leaf movement",
        "rainy city": "steady rain, wet pavement, distant traffic, low thunder",
        "city": "distant traffic, urban room tone, soft mechanical hum",
        "spaceship": "low engine hum, filtered air vents, subtle electronic pulses",
        "cave": "deep stone reverb, distant dripping water, low subterranean air",
        "ocean": "rolling waves, distant gulls, soft coastal wind",
        "desert": "dry wind, sparse sand movement, wide open silence",
        "room": "subtle interior room tone, distant muffled movement",
    }
    MOOD_TEXTURES = {
        "horror": "dark eerie cinematic tension, distant unsettling details",
        "sad": "melancholic atmospheric texture, restrained emotional weight",
        "cinematic": "wide cinematic atmosphere, slow evolving low texture",
        "fantasy": "magical shimmering aura, soft ethereal movement",
        "sci-fi": "futuristic synthetic ambience, clean low drones",
        "calm": "soft peaceful ambience, gentle natural movement",
        "neutral": "natural cinematic ambience, subtle environmental realism",
    }

    def analyze_scene(self, scene_data: Any) -> dict[str, Any]:
        text = self._scene_text(scene_data)
        mood = self._detect_from_keywords(text, self.MOOD_KEYWORDS, default="neutral")
        environment = self._detect_from_keywords(text, self.ENVIRONMENT_KEYWORDS, default="room")
        emotional_tone = self._emotional_tone_for_mood(mood)
        tension_level = self._detect_tension(text, mood)
        ambience_prompt = self._build_ambience_prompt(
            mood=mood,
            environment=environment,
            emotional_tone=emotional_tone,
            tension_level=tension_level,
        )
        return AudioSceneMetadata(
            mood=mood,
            environment=environment,
            emotional_tone=emotional_tone,
            tension_level=tension_level,
            ambience_prompt=ambience_prompt,
        ).to_dict()

    def analyze_scenes(self, scenes: list[Any]) -> list[dict[str, Any]]:
        return [self.analyze_scene(scene) for scene in scenes]

    def _scene_text(self, scene_data: Any) -> str:
        if isinstance(scene_data, str):
            return self._normalize_text(scene_data)
        if isinstance(scene_data, dict):
            parts = [
                scene_data.get("narration"),
                scene_data.get("image_prompt"),
                scene_data.get("caption_text"),
                scene_data.get("description"),
                scene_data.get("prompt"),
            ]
            return self._normalize_text(" ".join(str(part) for part in parts if part))

        parts = []
        for attribute in ("narration", "image_prompt", "caption_text", "description", "prompt"):
            value = getattr(scene_data, attribute, None)
            if value:
                parts.append(str(value))
        return self._normalize_text(" ".join(parts))

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()

    def _detect_from_keywords(
        self,
        text: str,
        keyword_map: dict[str, tuple[str, ...]],
        *,
        default: str,
    ) -> str:
        scored: list[tuple[int, str]] = []
        for label, keywords in keyword_map.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score:
                scored.append((score, label))
        if not scored:
            return default
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    def _detect_tension(self, text: str, mood: str) -> float:
        if any(keyword in text for keyword in self.TENSION_KEYWORDS["high"]):
            return 0.9
        if any(keyword in text for keyword in self.TENSION_KEYWORDS["medium"]):
            return 0.55
        if any(keyword in text for keyword in self.TENSION_KEYWORDS["low"]):
            return 0.2
        if mood == "horror":
            return 0.8
        if mood in {"cinematic", "sci-fi"}:
            return 0.55
        return 0.35

    def _emotional_tone_for_mood(self, mood: str) -> str:
        return {
            "horror": "fearful",
            "sad": "melancholic",
            "cinematic": "dramatic",
            "fantasy": "wonder",
            "sci-fi": "mysterious",
            "calm": "peaceful",
        }.get(mood, "neutral")

    def _build_ambience_prompt(
        self,
        *,
        mood: str,
        environment: str,
        emotional_tone: str,
        tension_level: float,
    ) -> str:
        environment_texture = self.ENVIRONMENT_TEXTURES.get(environment, self.ENVIRONMENT_TEXTURES["room"])
        mood_texture = self.MOOD_TEXTURES.get(mood, self.MOOD_TEXTURES["neutral"])
        intensity = "subtle" if tension_level < 0.4 else "tense" if tension_level < 0.75 else "high tension"
        return (
            f"{mood_texture}, {environment_texture}, {intensity} {emotional_tone} atmosphere, "
            "cinematic ambience loop, no vocals, no melody, clean background sound design"
        )
