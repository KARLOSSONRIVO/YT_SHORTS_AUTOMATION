from __future__ import annotations

import random
import re
from pathlib import Path


class MusicService:
    SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
    MOOD_FOLDER_MAP = {
        "ambient": "ambient",
        "neutral": "ambient",
        "calm": "ambient",
        "sad": "sad",
        "dramatic": "cinematic",
        "cinematic": "cinematic",
        "epic": "cinematic",
        "horror": "horror",
        "scary": "horror",
        "mysterious": "horror",
    }

    def __init__(self, music_assets_path: str) -> None:
        configured_root = Path(music_assets_path)
        if not configured_root.is_absolute():
            configured_root = Path.cwd() / configured_root

        app_assets_root = Path(__file__).resolve().parents[1] / "assets" / "music"
        if configured_root.exists():
            self.music_root = configured_root
        elif app_assets_root.exists():
            self.music_root = app_assets_root
        else:
            self.music_root = configured_root

    def list_available_tracks(self) -> dict[str, list[str]]:
        if not self.music_root.exists():
            return {}

        tracks: dict[str, list[str]] = {}
        for mood_dir in sorted(path for path in self.music_root.iterdir() if path.is_dir()):
            mood_tracks = [
                str(track.resolve())
                for track in sorted(mood_dir.iterdir())
                if track.is_file() and track.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ]
            if mood_tracks:
                tracks[mood_dir.name.lower()] = mood_tracks
        return tracks

    def get_music_for_mood(self, mood: str) -> str | None:
        tracks_by_mood = self.list_available_tracks()
        if not tracks_by_mood:
            return None

        normalized_mood = self._normalize_mood(mood)
        mood_folder = self.MOOD_FOLDER_MAP.get(normalized_mood, "ambient")
        choices = tracks_by_mood.get(mood_folder) or tracks_by_mood.get("ambient") or []
        if not choices:
            return None
        return random.choice(choices)

    def detect_mood(self, script: str) -> str:
        normalized_script = re.sub(r"\s+", " ", script.lower())
        if any(token in normalized_script for token in ("ghost", "terror", "dark", "fear", "haunted", "murder")):
            return "horror"
        if any(token in normalized_script for token in ("grief", "loss", "heartbreak", "tragedy", "tears", "lonely")):
            return "sad"
        if any(token in normalized_script for token in ("epic", "legend", "cinematic", "battle", "empire", "dramatic")):
            return "cinematic"
        return "neutral"

    def _normalize_mood(self, mood: str) -> str:
        cleaned = re.sub(r"[^a-z]+", " ", mood.lower()).strip()
        return cleaned or "neutral"
