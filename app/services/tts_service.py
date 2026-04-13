from pathlib import Path
import wave

from app.core.exceptions import IntegrationError
from app.integrations.huggingface_client import HuggingFaceClient
from app.schemas.faceless_video import AudioGenerationRequest, AudioGenerationResponse


class TTSService:
    WORDS_PER_SECOND = 2.35

    def __init__(
        self,
        *,
        huggingface_client: HuggingFaceClient,
        ffmpeg_client,
        output_dir: str,
        model: str,
        model_path: str | None = None,
        allow_placeholder_generation: bool = False,
    ) -> None:
        self.huggingface_client = huggingface_client
        self.ffmpeg_client = ffmpeg_client
        self.output_dir = Path(output_dir)
        self.model = model
        self.model_path = Path(model_path) if model_path else None
        self.allow_placeholder_generation = allow_placeholder_generation
        self._kokoro_pipelines: dict[str, object] = {}

    def generate_narration(self, payload: AudioGenerationRequest) -> AudioGenerationResponse:
        job_dir = self.output_dir / payload.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        output_path = job_dir / "narration.wav"
        duration = self._estimate_duration(payload)

        try:
            if self._has_local_kokoro():
                self._generate_local_kokoro(payload=payload, output_path=output_path)
            else:
                audio_content = self.huggingface_client.text_to_speech(
                    model=self.model,
                    text=payload.narration,
                    voice=payload.voice,
                )
                raw_path = job_dir / "narration_hf_audio"
                raw_path.write_bytes(audio_content)
                self._normalize_audio(raw_path=raw_path, output_path=output_path)
        except Exception as exc:
            if not self.allow_placeholder_generation:
                if isinstance(exc, IntegrationError):
                    raise
                raise IntegrationError(f"Hugging Face TTS generation failed: {exc}") from exc
            self._write_silent_audio(output_path=output_path, duration=duration)

        duration = self._audio_duration(output_path) or duration

        return AudioGenerationResponse(
            job_id=payload.job_id,
            project_id=payload.project_id,
            audio_path=str(output_path.resolve()),
            audio_url=f"/outputs/{payload.job_id}/{output_path.name}",
            duration_seconds=duration,
            voice=payload.voice,
        )

    def _has_local_kokoro(self) -> bool:
        if not self.model_path:
            return False
        return (self.model_path / "kokoro-v1_0.pth").exists() and (self.model_path / "config.json").exists()

    def _generate_local_kokoro(self, *, payload: AudioGenerationRequest, output_path: Path) -> None:
        try:
            import numpy as np
            import soundfile as sf
            from kokoro import KModel, KPipeline
        except ImportError as exc:
            raise IntegrationError(
                "Local Kokoro requires kokoro, soundfile, and their dependencies. Rebuild the Python Docker image."
            ) from exc

        voice = payload.voice or "af_sarah"
        voice_path = self._resolve_voice_path(voice)
        lang_code = self._lang_code_for_voice(voice)
        pipeline = self._kokoro_pipeline(lang_code=lang_code, kmodel_class=KModel, pipeline_class=KPipeline)
        chunks = []

        generator = pipeline(
            payload.narration,
            voice=str(voice_path),
            speed=payload.speaking_rate,
            split_pattern=r"\n+",
        )
        for _graphemes, _phonemes, audio in generator:
            if audio is None:
                continue
            chunks.append(np.asarray(audio, dtype=np.float32))

        if not chunks:
            raise IntegrationError("Local Kokoro did not produce audio.")

        audio = np.concatenate(chunks)
        sf.write(str(output_path), audio, 24000)

    def _resolve_voice_path(self, voice: str) -> Path:
        if not self.model_path:
            raise IntegrationError("PY_WORKER_TTS_MODEL_PATH is required for local Kokoro.")

        voice_name = voice.removesuffix(".pt")
        voice_path = self.model_path / "voices" / f"{voice_name}.pt"
        if voice_path.exists():
            return voice_path

        fallback_path = self.model_path / "voices" / "af_sarah.pt"
        if fallback_path.exists():
            return fallback_path

        raise IntegrationError(f"Kokoro voice file was not found for voice '{voice}'.")

    def _kokoro_pipeline(self, *, lang_code: str, kmodel_class, pipeline_class):
        if lang_code in self._kokoro_pipelines:
            return self._kokoro_pipelines[lang_code]
        if not self.model_path:
            raise IntegrationError("PY_WORKER_TTS_MODEL_PATH is required for local Kokoro.")

        model = kmodel_class(
            repo_id=self.model,
            config=str(self.model_path / "config.json"),
            model=str(self.model_path / "kokoro-v1_0.pth"),
        ).eval()
        pipeline = pipeline_class(
            lang_code=lang_code,
            repo_id=self.model,
            model=model,
            device="cpu",
        )
        self._kokoro_pipelines[lang_code] = pipeline
        return pipeline

    def _lang_code_for_voice(self, voice: str) -> str:
        voice_name = voice.removesuffix(".pt")
        if voice_name.startswith("b"):
            return "b"
        if voice_name.startswith("e"):
            return "e"
        if voice_name.startswith("f"):
            return "f"
        if voice_name.startswith("h"):
            return "h"
        if voice_name.startswith("i"):
            return "i"
        if voice_name.startswith("j"):
            return "j"
        if voice_name.startswith("p"):
            return "p"
        if voice_name.startswith("z"):
            return "z"
        return "a"

    def _normalize_audio(self, *, raw_path: Path, output_path: Path) -> None:
        if not self.ffmpeg_client.is_available():
            output_path.write_bytes(raw_path.read_bytes())
            return

        self.ffmpeg_client.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(raw_path),
                "-ar",
                "44100",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ]
        )

    def _write_silent_audio(self, *, output_path: Path, duration: float) -> None:
        if not self.ffmpeg_client.is_available():
            raise IntegrationError("ffmpeg is required to create placeholder narration audio.")

        self.ffmpeg_client.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=44100:cl=mono",
                "-t",
                str(duration),
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ]
        )

    def _estimate_duration(self, payload: AudioGenerationRequest) -> float:
        word_count = max(len(payload.narration.split()), 1)
        return max(round((word_count / self.WORDS_PER_SECOND) / payload.speaking_rate, 2), 3.0)

    def _audio_duration(self, audio_path: Path) -> float | None:
        try:
            with wave.open(str(audio_path), "rb") as audio_file:
                frame_count = audio_file.getnframes()
                frame_rate = audio_file.getframerate()
                if frame_rate <= 0:
                    return None
                return round(frame_count / float(frame_rate), 2)
        except wave.Error:
            return None
