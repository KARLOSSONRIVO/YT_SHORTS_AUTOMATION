from __future__ import annotations

import re
import wave
from pathlib import Path

from app.core.exceptions import IntegrationError
from app.integrations.gemini_client import GeminiClient
from app.schemas.faceless_video import (
    AudioGenerationRequest,
    AudioGenerationResponse,
    VoiceOption,
    VoicePreviewResponse,
)
from app.utils.output_paths import output_url, stage_output_dir


class TTSService:
    SAMPLE_RATE = 24_000
    CHANNELS = 1
    SAMPLE_WIDTH = 2
    VOICES = (
        ("Kore", "Kore", "English / multilingual", "Female", "Firm and clear"),
        ("Aoede", "Aoede", "English / multilingual", "Female", "Breezy and natural"),
        ("Leda", "Leda", "English / multilingual", "Female", "Youthful"),
        ("Zephyr", "Zephyr", "English / multilingual", "Female", "Bright"),
        ("Puck", "Puck", "English / multilingual", "Male", "Upbeat"),
        ("Charon", "Charon", "English / multilingual", "Male", "Informative"),
        ("Orus", "Orus", "English / multilingual", "Male", "Firm"),
        ("Fenrir", "Fenrir", "English / multilingual", "Male", "Excitable"),
    )
    LEGACY_VOICE_MAP = {
        "af_sarah": "Kore",
        "af_heart": "Aoede",
        "af_bella": "Leda",
        "am_adam": "Charon",
        "am_michael": "Orus",
        "bf_emma": "Zephyr",
        "bm_george": "Puck",
    }

    def __init__(
        self,
        *,
        gemini_client: GeminiClient,
        ffmpeg_client,
        output_dir: str,
        model: str,
        allow_placeholder_generation: bool = False,
    ) -> None:
        self.gemini_client = gemini_client
        self.ffmpeg_client = ffmpeg_client
        self.output_dir = Path(output_dir)
        self.model = model
        self.allow_placeholder_generation = allow_placeholder_generation

    def list_available_voices(self) -> list[VoiceOption]:
        sample = "In the last few months, this faceless channel has exploded."
        return [
            VoiceOption(
                voice=voice,
                label=label,
                language=language,
                gender=gender,
                quality_grade=style,
                sample_text=sample,
            )
            for voice, label, language, gender, style in self.VOICES
        ]

    def generate_narration(self, payload: AudioGenerationRequest) -> AudioGenerationResponse:
        stage_dir = stage_output_dir(
            output_dir=self.output_dir,
            output_bucket=payload.output_bucket,
            project_title=payload.project_title,
            project_id=payload.project_id,
            stage_name="audio",
        )
        stage_dir.mkdir(parents=True, exist_ok=True)
        output_path = stage_dir / "narration.wav"
        voice = self._resolve_voice(payload.voice)
        pace = self._pace_instruction(payload.speaking_rate)
        prompt = f"{pace} Read this narration exactly as written, without adding commentary:\n{payload.narration.strip()}"
        pcm = self.gemini_client.generate_speech(model=self.model, text=prompt, voice=voice)
        self._write_audio(output_path, pcm)
        return AudioGenerationResponse(
            job_id=payload.job_id,
            project_id=payload.project_id,
            audio_path=str(output_path.resolve()),
            audio_url=output_url(output_dir=self.output_dir, file_path=output_path),
            duration_seconds=self._duration(output_path),
            voice=voice,
        )

    def generate_voice_preview(self, voice: str, text: str | None = None) -> VoicePreviewResponse:
        resolved_voice = self._resolve_voice(voice)
        sample_text = (text or "In the last few months, this faceless channel has exploded.").strip()
        preview_dir = self.output_dir / "voice-previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        safe_voice = re.sub(r"[^a-z0-9_-]+", "-", resolved_voice.lower())
        output_path = preview_dir / f"{safe_voice}.wav"
        pcm = self.gemini_client.generate_speech(
            model=self.model,
            text=f"Read naturally and clearly: {sample_text}",
            voice=resolved_voice,
        )
        self._write_audio(output_path, pcm)
        return VoicePreviewResponse(
            voice=resolved_voice,
            audio_path=str(output_path.resolve()),
            audio_url=output_url(output_dir=self.output_dir, file_path=output_path),
            sample_text=sample_text,
        )

    def _resolve_voice(self, voice: str) -> str:
        candidate = self.LEGACY_VOICE_MAP.get(voice, voice)
        known = {item[0].lower(): item[0] for item in self.VOICES}
        resolved = known.get(candidate.lower())
        if resolved is None and re.fullmatch(r"[a-z]{2}_.+", voice.lower()):
            # Preserve projects saved with any Kokoro-style voice ID.
            resolved = "Kore" if voice.lower()[1] == "f" else "Charon"
        if resolved is None:
            raise IntegrationError(f"Unknown Gemini voice '{voice}'.")
        return resolved

    def _write_audio(self, output_path: Path, data: bytes) -> None:
        if data[:4] == b"RIFF":
            output_path.write_bytes(data)
            return
        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(self.CHANNELS)
            wav_file.setsampwidth(self.SAMPLE_WIDTH)
            wav_file.setframerate(self.SAMPLE_RATE)
            wav_file.writeframes(data)

    def _duration(self, output_path: Path) -> float:
        with wave.open(str(output_path), "rb") as wav_file:
            return round(wav_file.getnframes() / max(wav_file.getframerate(), 1), 2)

    def _pace_instruction(self, speaking_rate: float) -> str:
        if speaking_rate < 0.9:
            return "Speak at a measured, slightly slow storytelling pace."
        if speaking_rate > 1.1:
            return "Speak at a brisk, energetic pace while remaining clear."
        return "Speak at a natural conversational storytelling pace."
