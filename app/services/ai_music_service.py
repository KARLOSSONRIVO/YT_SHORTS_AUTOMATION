from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any

from app.core.exceptions import IntegrationError

logger = logging.getLogger(__name__)


class AIMusicService:
    MODEL_REPOSITORIES = {
        "musicgen-small": "facebook/musicgen-small",
        "musicgen-medium": "facebook/musicgen-medium",
        "musicgen-large": "facebook/musicgen-large",
        "musicgen-stereo-small": "facebook/musicgen-stereo-small",
        "musicgen-stereo-medium": "facebook/musicgen-stereo-medium",
        "musicgen-stereo-large": "facebook/musicgen-stereo-large",
    }
    SUPPORTED_OUTPUT_FORMATS = {"wav", "mp3"}

    def __init__(
        self,
        *,
        huggingface_client,
        ffmpeg_client=None,
        cache_dir: str = "assets/generated_music",
        model: str = "musicgen-small",
        enabled: bool = False,
        default_duration_seconds: float = 30.0,
        allow_placeholder_generation: bool = False,
    ) -> None:
        self.huggingface_client = huggingface_client
        self.ffmpeg_client = ffmpeg_client
        self.cache_dir = self._resolve_cache_dir(cache_dir)
        self.model = model
        self.enabled = enabled
        self.default_duration_seconds = default_duration_seconds
        self.allow_placeholder_generation = allow_placeholder_generation

    def generate_music(
        self,
        *,
        scenes: list[Any] | None = None,
        prompt: str | None = None,
        mood: str | None = None,
        duration_seconds: float | None = None,
        output_format: str = "wav",
    ) -> dict[str, Any]:
        output_format = output_format.lower().lstrip(".")
        if output_format not in self.SUPPORTED_OUTPUT_FORMATS:
            raise IntegrationError(f"Unsupported music output format '{output_format}'.")

        music_prompt = self.build_music_prompt(scenes=scenes or [], prompt=prompt, mood=mood)
        duration = self._generation_duration(duration_seconds)
        cache_key = self._cache_key(prompt=music_prompt, duration_seconds=duration, model=self.model)
        cached_path = self.cache_music(prompt=music_prompt, duration_seconds=duration, output_format=output_format)
        if cached_path.exists():
            return self._result_payload(
                music_path=cached_path,
                prompt=music_prompt,
                duration_seconds=duration,
                cache_key=cache_key,
                cached=True,
            )

        wav_path = self.cache_music(prompt=music_prompt, duration_seconds=duration, output_format="wav")
        if output_format == "mp3" and wav_path.exists():
            self._export_music(wav_path, cached_path, output_format="mp3")
            return self._result_payload(
                music_path=cached_path,
                prompt=music_prompt,
                duration_seconds=duration,
                cache_key=cache_key,
                cached=True,
            )

        if not self.enabled:
            raise IntegrationError("AI music generation is disabled. Set PY_WORKER_ENABLE_AI_MUSIC=true.")

        try:
            self._generate_music(prompt=music_prompt, output_path=wav_path, duration_seconds=duration)
        except Exception as exc:
            if not self.allow_placeholder_generation:
                if isinstance(exc, IntegrationError):
                    raise
                raise IntegrationError(f"AI music generation failed: {exc}") from exc
            logger.warning("AI music generation failed; writing placeholder music.", exc_info=exc)
            self._write_placeholder_music(wav_path, duration_seconds=duration)

        final_path = wav_path
        if output_format == "mp3":
            final_path = self.cache_music(prompt=music_prompt, duration_seconds=duration, output_format="mp3")
            self._export_music(wav_path, final_path, output_format="mp3")

        self._write_metadata(
            music_path=final_path,
            prompt=music_prompt,
            duration_seconds=duration,
            cache_key=cache_key,
        )
        return self._result_payload(
            music_path=final_path,
            prompt=music_prompt,
            duration_seconds=duration,
            cache_key=cache_key,
            cached=False,
        )

    def build_music_prompt(
        self,
        *,
        scenes: list[Any],
        prompt: str | None = None,
        mood: str | None = None,
    ) -> str:
        if prompt and prompt.strip():
            base_prompt = prompt.strip()
        else:
            scene_text = self._scenes_text(scenes)
            mood_text = (mood or self._detect_mood(scene_text)).strip()
            base_prompt = f"{mood_text} cinematic background music for a vertical story video"

        prompt_parts = [
            " ".join(base_prompt.split()),
            "instrumental",
            "cinematic underscore",
            "background music under narration",
            "no vocals",
            "no singing",
            "no spoken words",
            "smooth loopable ending",
        ]
        return ", ".join(part for part in prompt_parts if part)

    def cache_music(
        self,
        *,
        prompt: str,
        duration_seconds: float | None = None,
        output_format: str = "wav",
        generated_path: Path | None = None,
    ) -> Path:
        duration = self._generation_duration(duration_seconds)
        extension = output_format.lower().lstrip(".")
        cache_key = self._cache_key(prompt=prompt, duration_seconds=duration, model=self.model)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        target_path = self.cache_dir / f"{cache_key}.{extension}"
        if generated_path is not None and generated_path.exists() and generated_path.resolve() != target_path.resolve():
            shutil.copyfile(generated_path, target_path)
        return target_path

    def _generate_music(self, *, prompt: str, output_path: Path, duration_seconds: float) -> None:
        if not self.huggingface_client.is_configured():
            raise IntegrationError("PY_WORKER_HF_TOKEN is required for AI music generation.")

        audio_bytes = self.huggingface_client.text_to_audio(
            model=self._model_repository(),
            prompt=prompt,
            parameters={
                "duration": duration_seconds,
                "max_new_tokens": int(duration_seconds * 50),
            },
        )
        output_path.write_bytes(audio_bytes)

    def _export_music(self, input_path: Path, output_path: Path, *, output_format: str) -> None:
        if output_format == "wav":
            shutil.copyfile(input_path, output_path)
            return
        if self.ffmpeg_client is None or not self.ffmpeg_client.is_available():
            raise IntegrationError("ffmpeg is required to export generated music as mp3.")
        self.ffmpeg_client.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(input_path),
                "-codec:a",
                "libmp3lame",
                "-b:a",
                "192k",
                str(output_path),
            ]
        )

    def _write_placeholder_music(self, output_path: Path, *, duration_seconds: float) -> None:
        if self.ffmpeg_client is None or not self.ffmpeg_client.is_available():
            raise IntegrationError("ffmpeg is required to create placeholder music.")
        self.ffmpeg_client.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency=220:duration={duration_seconds}",
                "-ar",
                "44100",
                "-ac",
                "2",
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ]
        )

    def _write_metadata(
        self,
        *,
        music_path: Path,
        prompt: str,
        duration_seconds: float,
        cache_key: str,
    ) -> None:
        metadata_path = music_path.with_suffix(".json")
        metadata_path.write_text(
            json.dumps(
                {
                    "cache_key": cache_key,
                    "model": self.model,
                    "prompt": prompt,
                    "duration_seconds": duration_seconds,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _result_payload(
        self,
        *,
        music_path: Path,
        prompt: str,
        duration_seconds: float,
        cache_key: str,
        cached: bool,
    ) -> dict[str, Any]:
        return {
            "music_path": str(music_path.resolve()),
            "prompt": prompt,
            "duration_seconds": duration_seconds,
            "cache_key": cache_key,
            "cached": cached,
        }

    def _generation_duration(self, duration_seconds: float | None) -> float:
        duration = float(duration_seconds or self.default_duration_seconds)
        return round(min(max(duration, 5.0), 120.0), 2)

    def _cache_key(self, *, prompt: str, duration_seconds: float, model: str) -> str:
        normalized = json.dumps(
            {
                "model": model,
                "duration_seconds": duration_seconds,
                "prompt": " ".join(prompt.lower().split()),
            },
            sort_keys=True,
        )
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]

    def _resolve_cache_dir(self, cache_dir: str) -> Path:
        path = Path(cache_dir)
        if path.is_absolute():
            return path
        return Path.cwd() / path

    def _model_repository(self) -> str:
        return self.MODEL_REPOSITORIES.get(self.model.lower(), self.model)

    def _scenes_text(self, scenes: list[Any]) -> str:
        parts: list[str] = []
        for scene in scenes:
            if isinstance(scene, dict):
                parts.extend(str(scene.get(key) or "") for key in ("narration", "image_prompt", "caption_text"))
                continue
            parts.extend(
                str(getattr(scene, key, "") or "")
                for key in ("narration", "image_prompt", "caption_text")
            )
        return " ".join(part for part in parts if part).lower()

    def _detect_mood(self, text: str) -> str:
        if any(token in text for token in ("ghost", "terror", "dark", "fear", "haunted", "murder")):
            return "horror tension"
        if any(token in text for token in ("grief", "loss", "heartbreak", "tragedy", "tears", "lonely")):
            return "sad emotional"
        if any(token in text for token in ("epic", "legend", "battle", "empire", "dramatic")):
            return "epic cinematic"
        if any(token in text for token in ("magic", "dragon", "kingdom", "portal", "enchanted")):
            return "fantasy wonder"
        return "subtle atmospheric"
