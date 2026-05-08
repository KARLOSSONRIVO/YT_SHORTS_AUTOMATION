from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

from app.core.exceptions import IntegrationError
from app.services.audio_scene_analysis_service import AudioSceneAnalysisService

logger = logging.getLogger(__name__)


class AIAudioService:
    MODEL_REPOSITORIES = {
        "stable-audio-open": "stabilityai/stable-audio-open-1.0",
        "stable-audio-open-1.0": "stabilityai/stable-audio-open-1.0",
        "audioldm2": "cvssp/audioldm2",
        "audioldm2-large": "cvssp/audioldm2-large",
        "audioldm2-music": "cvssp/audioldm2-music",
    }
    SUPPORTED_OUTPUT_FORMATS = {"wav", "mp3"}

    def __init__(
        self,
        *,
        huggingface_client=None,
        ffmpeg_client=None,
        scene_analysis_service: AudioSceneAnalysisService | None = None,
        cache_dir: str = "assets/generated_audio",
        model: str = "stable-audio-open",
        enabled: bool = False,
        ambience_volume: float = 0.08,
        default_duration_seconds: float = 12.0,
        inference_steps: int = 100,
        allow_placeholder_generation: bool = False,
    ) -> None:
        self.huggingface_client = huggingface_client
        self.ffmpeg_client = ffmpeg_client
        self.scene_analysis_service = scene_analysis_service or AudioSceneAnalysisService()
        self.cache_dir = self._resolve_cache_dir(cache_dir)
        self.model = model
        self.enabled = enabled
        self.ambience_volume = min(max(ambience_volume, 0.0), 1.0)
        self.default_duration_seconds = default_duration_seconds
        self.inference_steps = inference_steps
        self.allow_placeholder_generation = allow_placeholder_generation

    def generate_ambience(
        self,
        scene_data: Any,
        *,
        duration_seconds: float | None = None,
        output_format: str = "wav",
    ) -> dict[str, Any]:
        output_format = output_format.lower().lstrip(".")
        if output_format not in self.SUPPORTED_OUTPUT_FORMATS:
            raise IntegrationError(f"Unsupported ambience output format '{output_format}'.")

        metadata = self.scene_analysis_service.analyze_scene(scene_data)
        prompt = self.build_audio_prompt({**self._scene_dict(scene_data), **metadata})
        duration = self._generation_duration(duration_seconds, scene_data)
        cache_key = self._cache_key(prompt=prompt, duration_seconds=duration, model=self.model)
        cached_path = self.cache_audio(prompt=prompt, duration_seconds=duration, output_format=output_format)
        if cached_path.exists():
            return self._result_payload(
                audio_path=cached_path,
                prompt=prompt,
                metadata=metadata,
                duration_seconds=duration,
                cache_key=cache_key,
                cached=True,
            )

        wav_path = self.cache_audio(prompt=prompt, duration_seconds=duration, output_format="wav")
        if output_format == "mp3" and wav_path.exists():
            self._export_audio(wav_path, cached_path, output_format="mp3")
            return self._result_payload(
                audio_path=cached_path,
                prompt=prompt,
                metadata=metadata,
                duration_seconds=duration,
                cache_key=cache_key,
                cached=True,
            )

        if not self.enabled:
            raise IntegrationError("AI ambience generation is disabled. Set PY_WORKER_ENABLE_AI_AMBIENCE=true.")

        try:
            self._generate_audio(prompt=prompt, output_path=wav_path, duration_seconds=duration)
        except Exception as exc:
            if not self.allow_placeholder_generation:
                if isinstance(exc, IntegrationError):
                    raise
                raise IntegrationError(f"AI ambience generation failed: {exc}") from exc
            logger.warning("AI ambience generation failed; writing placeholder ambience.", exc_info=exc)
            self._write_placeholder_ambience(wav_path, duration_seconds=duration)

        final_path = wav_path
        if output_format == "mp3":
            final_path = self.cache_audio(prompt=prompt, duration_seconds=duration, output_format="mp3")
            self._export_audio(wav_path, final_path, output_format="mp3")

        self._write_metadata(
            audio_path=final_path,
            prompt=prompt,
            metadata=metadata,
            duration_seconds=duration,
            cache_key=cache_key,
        )
        return self._result_payload(
            audio_path=final_path,
            prompt=prompt,
            metadata=metadata,
            duration_seconds=duration,
            cache_key=cache_key,
            cached=False,
        )

    def build_audio_prompt(self, scene_data: Any) -> str:
        data = self._scene_dict(scene_data)
        ambience_prompt = str(data.get("ambience_prompt") or "").strip()
        if ambience_prompt:
            base_prompt = ambience_prompt
        else:
            metadata = self.scene_analysis_service.analyze_scene(scene_data)
            base_prompt = str(metadata["ambience_prompt"])

        normalized = " ".join(base_prompt.split())
        prompt_parts = [
            normalized,
            "10 to 20 second seamless ambience bed",
            "professional film sound design",
            "low volume background layer under narration",
            "no vocals",
            "no spoken words",
            "no lead melody",
        ]
        return ", ".join(part for part in prompt_parts if part)

    def cache_audio(
        self,
        *,
        prompt: str,
        duration_seconds: float | None = None,
        output_format: str = "wav",
        generated_path: Path | None = None,
    ) -> Path:
        duration = self._generation_duration(duration_seconds, None)
        extension = output_format.lower().lstrip(".")
        cache_key = self._cache_key(prompt=prompt, duration_seconds=duration, model=self.model)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        target_path = self.cache_dir / f"{cache_key}.{extension}"
        if generated_path is not None and generated_path.exists() and generated_path.resolve() != target_path.resolve():
            shutil.copyfile(generated_path, target_path)
        return target_path

    def cleanup_audio(
        self,
        *,
        prompt: str | None = None,
        max_age_seconds: float | None = None,
    ) -> list[str]:
        if not self.cache_dir.exists():
            return []

        deleted: list[str] = []
        expected_prefix = None
        if prompt:
            expected_prefix = self._cache_key(
                prompt=prompt,
                duration_seconds=self._generation_duration(None, None),
                model=self.model,
            )

        for path in self.cache_dir.iterdir():
            if not path.is_file():
                continue
            if expected_prefix and not path.name.startswith(expected_prefix):
                continue
            if max_age_seconds is not None:
                age_seconds = max(0.0, time.time() - path.stat().st_mtime)
                if age_seconds < max_age_seconds:
                    continue
            path.unlink(missing_ok=True)
            deleted.append(str(path.resolve()))
        return deleted

    def _generate_audio(self, *, prompt: str, output_path: Path, duration_seconds: float) -> None:
        if self.huggingface_client is None:
            raise IntegrationError("Hugging Face client is required for AI ambience generation.")
        if not self.huggingface_client.is_configured():
            raise IntegrationError("PY_WORKER_HF_TOKEN is required for AI ambience generation.")

        audio_bytes = self.huggingface_client.text_to_audio(
            model=self._model_repository(),
            prompt=prompt,
            parameters={
                "duration": duration_seconds,
                "audio_length_in_s": duration_seconds,
                "num_inference_steps": self.inference_steps,
                "negative_prompt": "low quality, distorted, clipping, vocals, speech, melody",
            },
        )
        output_path.write_bytes(audio_bytes)

    def _export_audio(self, input_path: Path, output_path: Path, *, output_format: str) -> None:
        if output_format == "wav":
            shutil.copyfile(input_path, output_path)
            return
        if self.ffmpeg_client is None or not self.ffmpeg_client.is_available():
            raise IntegrationError("ffmpeg is required to export generated ambience as mp3.")
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

    def _write_placeholder_ambience(self, output_path: Path, *, duration_seconds: float) -> None:
        if self.ffmpeg_client is None or not self.ffmpeg_client.is_available():
            raise IntegrationError("ffmpeg is required to create placeholder ambience.")
        self.ffmpeg_client.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"anoisesrc=color=pink:amplitude=0.035:duration={duration_seconds}",
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
        audio_path: Path,
        prompt: str,
        metadata: dict[str, Any],
        duration_seconds: float,
        cache_key: str,
    ) -> None:
        metadata_path = audio_path.with_suffix(".json")
        metadata_path.write_text(
            json.dumps(
                {
                    "cache_key": cache_key,
                    "model": self.model,
                    "prompt": prompt,
                    "duration_seconds": duration_seconds,
                    "metadata": metadata,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _result_payload(
        self,
        *,
        audio_path: Path,
        prompt: str,
        metadata: dict[str, Any],
        duration_seconds: float,
        cache_key: str,
        cached: bool,
    ) -> dict[str, Any]:
        return {
            "audio_path": str(audio_path.resolve()),
            "prompt": prompt,
            "duration_seconds": duration_seconds,
            "cache_key": cache_key,
            "cached": cached,
            **metadata,
        }

    def _generation_duration(self, duration_seconds: float | None, scene_data: Any) -> float:
        if duration_seconds is None and scene_data is not None:
            duration_seconds = getattr(scene_data, "duration_seconds", None)
            if duration_seconds is None and isinstance(scene_data, dict):
                duration_seconds = scene_data.get("duration_seconds")
        duration = float(duration_seconds or self.default_duration_seconds)
        return round(min(max(duration, 10.0), 20.0), 2)

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

    def _scene_dict(self, scene_data: Any) -> dict[str, Any]:
        if isinstance(scene_data, dict):
            return dict(scene_data)
        if isinstance(scene_data, str):
            return {"description": scene_data}
        data: dict[str, Any] = {}
        for attribute in ("scene_index", "narration", "image_prompt", "caption_text", "duration_seconds"):
            value = getattr(scene_data, attribute, None)
            if value is not None:
                data[attribute] = value
        return data
