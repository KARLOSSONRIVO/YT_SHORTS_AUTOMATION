from __future__ import annotations

from array import array
from io import BytesIO
import logging
import math
import re
import wave
import zlib
from pathlib import Path

from app.core.exceptions import IntegrationError, ProviderRateLimitError
from app.integrations.gemini_client import GeminiClient
from app.integrations.pollinations_client import PollinationsClient
from app.schemas.faceless_video import AudioGenerationRequest, AudioGenerationResponse
from app.utils.output_paths import output_url, stage_output_dir

logger = logging.getLogger(__name__)


class TTSService:
    SAMPLE_RATE = 24_000
    CHANNELS = 1
    SAMPLE_WIDTH = 2
    POLLINATIONS_TTS_MAX_CHARS = 600
    POLLINATIONS_ELEVENLABS_MAX_CHARS = 3_800
    POLLINATIONS_CHUNK_PAUSE_MS = 120
    POLLINATIONS_TARGET_RMS_DBFS = -19.0
    POLLINATIONS_TARGET_PEAK_DBFS = -1.5
    VOICES = (
        ("Kore", "Kore", "English / multilingual", "Female", "Firm and clear"),
        ("Aoede", "Aoede", "English / multilingual", "Female", "Breezy and natural"),
        ("Leda", "Leda", "English / multilingual", "Female", "Youthful"),
        ("Zephyr", "Zephyr", "English / multilingual", "Female", "Bright"),
        ("Puck", "Puck", "English / multilingual", "Male", "Upbeat"),
        ("Charon", "Charon", "English / multilingual", "Male", "Informative"),
        ("Orus", "Orus", "English / multilingual", "Male", "Firm"),
        ("Fenrir", "Fenrir", "English / multilingual", "Male", "Excitable"),
        ("Callirrhoe", "Callirrhoe", "English / multilingual", "Female", "Easy-going"),
        ("Autonoe", "Autonoe", "English / multilingual", "Female", "Bright"),
        ("Enceladus", "Enceladus", "English / multilingual", "Male", "Breathy"),
        ("Iapetus", "Iapetus", "English / multilingual", "Male", "Clear"),
        ("Umbriel", "Umbriel", "English / multilingual", "Male", "Easy-going"),
        ("Algieba", "Algieba", "English / multilingual", "Male", "Smooth"),
        ("Despina", "Despina", "English / multilingual", "Female", "Smooth"),
        ("Erinome", "Erinome", "English / multilingual", "Female", "Clear"),
        ("Algenib", "Algenib", "English / multilingual", "Male", "Gravelly"),
        ("Rasalgethi", "Rasalgethi", "English / multilingual", "Male", "Informative"),
        ("Laomedeia", "Laomedeia", "English / multilingual", "Female", "Upbeat"),
        ("Achernar", "Achernar", "English / multilingual", "Female", "Soft"),
        ("Alnilam", "Alnilam", "English / multilingual", "Male", "Firm"),
        ("Schedar", "Schedar", "English / multilingual", "Male", "Even"),
        ("Gacrux", "Gacrux", "English / multilingual", "Female", "Mature"),
        ("Pulcherrima", "Pulcherrima", "English / multilingual", "Female", "Forward"),
        ("Achird", "Achird", "English / multilingual", "Male", "Friendly"),
        ("Zubenelgenubi", "Zubenelgenubi", "English / multilingual", "Male", "Casual"),
        ("Vindemiatrix", "Vindemiatrix", "English / multilingual", "Female", "Gentle"),
        ("Sadachbia", "Sadachbia", "English / multilingual", "Male", "Lively"),
        ("Sadaltager", "Sadaltager", "English / multilingual", "Male", "Knowledgeable"),
        ("Sulafat", "Sulafat", "English / multilingual", "Female", "Warm"),
    )
    SCRIPT_TONES = (
        (
            "somber and empathetic, with restrained sadness",
            (
                "death",
                "died",
                "funeral",
                "goodbye",
                "grief",
                "heartbreak",
                "loss",
                "painful",
                "regret",
                "tears",
            ),
        ),
        (
            "tense and suspenseful, building urgency around each revelation",
            (
                "danger",
                "discovered",
                "hidden",
                "secret",
                "suddenly",
                "truth",
                "until",
                "vanished",
            ),
        ),
        (
            "emotionally charged and serious without shouting",
            (
                "angry",
                "argument",
                "betrayal",
                "betrayed",
                "cheated",
                "fight",
                "furious",
                "kissed someone else",
                "lied",
                "revenge",
            ),
        ),
        (
            "warm and hopeful, becoming genuinely uplifting",
            (
                "finally",
                "hope",
                "joy",
                "overcame",
                "reunited",
                "survived",
                "triumph",
                "victory",
            ),
        ),
    )

    def __init__(
        self,
        *,
        gemini_client: GeminiClient,
        pollinations_client: PollinationsClient,
        ffmpeg_client,
        output_dir: str,
        model: str,
        fallback_model: str,
        allow_placeholder_generation: bool = False,
    ) -> None:
        self.gemini_client = gemini_client
        self.pollinations_client = pollinations_client
        self.ffmpeg_client = ffmpeg_client
        self.output_dir = Path(output_dir)
        self.model = model
        self.fallback_model = fallback_model
        self.allow_placeholder_generation = allow_placeholder_generation

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
        response_voice = voice
        pace = self._pace_instruction(payload.speaking_rate)
        narration = payload.narration.strip()
        prompt = f"{pace} Read this narration exactly as written, without adding commentary:\n{narration}"
        try:
            audio = self.gemini_client.generate_speech(
                model=self.model,
                text=prompt,
                voice=voice,
            )
        except ProviderRateLimitError:
            if (
                not self.pollinations_client.is_configured()
                or not self.fallback_model.strip()
            ):
                raise
            logger.warning(
                "Gemini speech generation is rate-limited; using Pollinations "
                "model %s.",
                self.fallback_model,
            )
            audio = self._generate_pollinations_narration(
                narration=narration,
                pace=pace,
                voice=voice,
                speaking_rate=payload.speaking_rate,
            )
            response_voice = self._pollinations_voice(narration)

        self._write_audio(output_path, audio)
        return AudioGenerationResponse(
            job_id=payload.job_id,
            project_id=payload.project_id,
            audio_path=str(output_path.resolve()),
            audio_url=output_url(output_dir=self.output_dir, file_path=output_path),
            duration_seconds=self._duration(output_path),
            voice=response_voice,
        )

    def _resolve_voice(self, voice: str) -> str:
        known = {item[0].lower(): item[0] for item in self.VOICES}
        resolved = known.get(voice.lower())
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

    def _fallback_instruction(
        self,
        *,
        pace: str,
        narration: str,
        narrator_profile: str,
        section_direction: str,
    ) -> str:
        tone = self._script_tone(narration)
        return (
            f"{pace} Use {narrator_profile}. Keep the delivery {tone}. "
            "Project every line clearly; never sound breathy, timid, weak, or "
            f"underpowered. {section_direction} Shape the emotion naturally to "
            "the meaning while staying believable, controlled, and cinematic."
        )

    @staticmethod
    def _fallback_voice_profile(narration: str) -> str:
        normalized = narration.lower()
        if any(
            signal in normalized
            for signal in (
                "anxiety",
                "brain",
                "choice",
                "clarity",
                "decide",
                "decision",
                "fear",
                "habit",
                "mind",
                "psychology",
            )
        ):
            return (
                "a mature, resonant storyteller with a warm lower register, "
                "firm projection, crisp diction, and calm authority"
            )
        if any(
            signal in normalized
            for signal in (
                "army",
                "battle",
                "empire",
                "king",
                "queen",
                "war",
                "warrior",
            )
        ):
            return (
                "a deep, commanding cinematic storyteller with mature weight, "
                "firm projection, and vivid dramatic presence"
            )
        return (
            "a mature, resonant cinematic storyteller with firm projection, "
            "crisp diction, warmth, and confident authority"
        )

    def _pollinations_voice(self, narration: str) -> str:
        if self.fallback_model.strip().lower().startswith("eleven"):
            return self._elevenlabs_voice(narration)
        return "alloy"

    @staticmethod
    def _elevenlabs_voice(narration: str) -> str:
        normalized = narration.lower()
        if any(
            signal in normalized
            for signal in (
                "anxiety",
                "brain",
                "choice",
                "clarity",
                "decide",
                "decision",
                "fear",
                "habit",
                "mind",
                "psychology",
            )
        ):
            return "brian"
        if any(
            signal in normalized
            for signal in (
                "army",
                "battle",
                "empire",
                "history",
                "king",
                "queen",
                "war",
                "warrior",
            )
        ):
            return "george"
        if any(
            signal in normalized
            for signal in (
                "angry",
                "argument",
                "betrayal",
                "betrayed",
                "cheated",
                "fight",
                "furious",
                "kissed someone else",
                "lied",
                "revenge",
            )
        ):
            return "domi"
        if any(
            signal in normalized
            for signal in (
                "death",
                "died",
                "funeral",
                "goodbye",
                "grief",
                "heartbreak",
                "loss",
                "painful",
                "regret",
                "tears",
            )
        ):
            return "james"
        return "brian"

    @staticmethod
    def _elevenlabs_opening_tag(narration: str) -> str:
        normalized = narration.lower()
        if any(signal in normalized for signal in ("death", "grief", "loss", "tears")):
            return "sorrowful"
        if any(
            signal in normalized
            for signal in ("angry", "betrayal", "cheated", "fight", "lied")
        ):
            return "frustrated"
        if any(
            signal in normalized
            for signal in ("hope", "joy", "overcame", "triumph", "victory")
        ):
            return "excited"
        if any(
            signal in normalized
            for signal in (
                "anxiety",
                "brain",
                "choice",
                "decide",
                "decision",
                "discovered",
                "fear",
                "hidden",
                "psychology",
                "secret",
            )
        ):
            return "curious"
        return "calm"

    def _elevenlabs_directed_text(self, text: str) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        if not sentences or not sentences[0]:
            return text.strip()
        directed = list(sentences)
        directed[0] = f"[{self._elevenlabs_opening_tag(text)}] {directed[0]}"
        if len(directed) >= 3:
            pivot_index = len(directed) // 2
            directed[pivot_index] = f"[pause] {directed[pivot_index]}"
        if len(directed) >= 2:
            directed[-1] = f"[determined] {directed[-1]}"
        return " ".join(directed)

    @staticmethod
    def _section_direction(index: int, total: int) -> str:
        if total == 1:
            return (
                "Open with immediate conviction and command attention, then "
                "build toward a decisive conclusion with confidence and weight."
            )
        if index == 0:
            return (
                "Open with immediate conviction and command attention from the "
                "first word; make the hook urgent and compelling."
            )
        if index == total - 1:
            return (
                "Build toward a decisive conclusion; land the final thought "
                "with confidence, purpose, and weight."
            )
        return (
            "Explain with grounded authority and empathy, increasing tension "
            "around each reveal without losing momentum."
        )

    def _script_tone(self, narration: str) -> str:
        normalized = narration.lower()
        ranked = [
            (sum(normalized.count(signal) for signal in signals), index, tone)
            for index, (tone, signals) in enumerate(self.SCRIPT_TONES)
        ]
        score, _, tone = max(ranked, key=lambda item: (item[0], -item[1]))
        if score:
            return tone
        if narration.count("!") >= 2:
            return "animated and energetic"
        if narration.count("?") >= 2:
            return "reflective and inquisitive"
        return "natural and emotionally attentive"

    def _generate_pollinations_narration(
        self,
        *,
        narration: str,
        pace: str,
        voice: str,
        speaking_rate: float,
    ) -> bytes:
        using_elevenlabs = self.fallback_model.strip().lower().startswith("eleven")
        max_chars = (
            self.POLLINATIONS_ELEVENLABS_MAX_CHARS
            if using_elevenlabs
            else self.POLLINATIONS_TTS_MAX_CHARS
        )
        chunks = self._narration_chunks(narration, max_chars=max_chars)
        narrator_profile = self._fallback_voice_profile(narration)
        fallback_voice = self._pollinations_voice(narration)
        seed = zlib.crc32(
            f"{self.fallback_model.lower()}|{fallback_voice}".encode("utf-8")
        )
        audio_chunks = []
        for index, chunk in enumerate(chunks):
            instruction = None
            request_text = chunk
            if using_elevenlabs:
                request_text = self._elevenlabs_directed_text(chunk)
            else:
                instruction = self._fallback_instruction(
                    pace=pace,
                    narration=narration,
                    narrator_profile=narrator_profile,
                    section_direction=self._section_direction(index, len(chunks)),
                )
            audio_chunks.append(
                self.pollinations_client.generate_speech(
                    model=self.fallback_model,
                    text=request_text,
                    voice=fallback_voice,
                    instruct=instruction,
                    speed=speaking_rate,
                    seed=seed,
                )
            )

        joined = self._join_wav_chunks(audio_chunks)
        return self._normalize_wav_loudness(
            joined,
            target_rms_dbfs=self.POLLINATIONS_TARGET_RMS_DBFS,
            target_peak_dbfs=self.POLLINATIONS_TARGET_PEAK_DBFS,
        )

    def _narration_chunks(
        self,
        narration: str,
        *,
        max_chars: int | None = None,
    ) -> list[str]:
        limit = max_chars or self.POLLINATIONS_TTS_MAX_CHARS
        sentences = re.split(r"(?<=[.!?])\s+", narration.strip())
        chunks: list[str] = []
        current = ""

        for sentence in sentences:
            remaining = sentence.strip()
            while remaining:
                available = limit - len(current)
                separator = 1 if current else 0
                if len(remaining) + separator <= available:
                    current = f"{current} {remaining}".strip()
                    remaining = ""
                    continue

                if current:
                    chunks.append(current)
                    current = ""
                    continue

                split_at = remaining.rfind(
                    " ",
                    0,
                    limit + 1,
                )
                if split_at <= 0:
                    split_at = limit
                chunks.append(remaining[:split_at].strip())
                remaining = remaining[split_at:].strip()

        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _join_wav_chunks(chunks: list[bytes]) -> bytes:
        if len(chunks) == 1:
            return chunks[0]

        reference: tuple[int, int, int, str] | None = None
        frames: list[bytes] = []
        try:
            for chunk in chunks:
                with wave.open(BytesIO(chunk), "rb") as wav_file:
                    signature = (
                        wav_file.getnchannels(),
                        wav_file.getsampwidth(),
                        wav_file.getframerate(),
                        wav_file.getcomptype(),
                    )
                    if reference is None:
                        reference = signature
                    elif signature != reference:
                        raise IntegrationError(
                            "Pollinations returned incompatible WAV chunks."
                        )
                    frames.append(wav_file.readframes(wav_file.getnframes()))
        except (EOFError, wave.Error) as exc:
            raise IntegrationError(
                "Pollinations returned an invalid WAV speech chunk."
            ) from exc

        if reference is None:
            raise IntegrationError("Pollinations returned no WAV speech chunks.")

        output = BytesIO()
        pause_frames = round(
            reference[2] * TTSService.POLLINATIONS_CHUNK_PAUSE_MS / 1_000
        )
        pause = b"\x00" * (pause_frames * reference[0] * reference[1])
        with wave.open(output, "wb") as wav_file:
            wav_file.setnchannels(reference[0])
            wav_file.setsampwidth(reference[1])
            wav_file.setframerate(reference[2])
            wav_file.setcomptype(reference[3], "not compressed")
            wav_file.writeframes(pause.join(frames))
        return output.getvalue()

    @staticmethod
    def _normalize_wav_loudness(
        data: bytes,
        *,
        target_rms_dbfs: float,
        target_peak_dbfs: float,
    ) -> bytes:
        try:
            with wave.open(BytesIO(data), "rb") as wav_file:
                params = wav_file.getparams()
                if params.sampwidth != 2 or params.comptype != "NONE":
                    raise IntegrationError(
                        "Pollinations returned an unsupported WAV speech format."
                    )
                samples = array("h", wav_file.readframes(params.nframes))
        except (EOFError, wave.Error) as exc:
            raise IntegrationError(
                "Pollinations returned invalid WAV speech audio."
            ) from exc

        if not samples:
            return data
        rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
        peak = max(abs(sample) for sample in samples)
        if rms == 0 or peak == 0:
            return data

        full_scale = 32_767
        target_rms = full_scale * 10 ** (target_rms_dbfs / 20)
        target_peak = full_scale * 10 ** (target_peak_dbfs / 20)
        gain = min(target_rms / rms, target_peak / peak)
        normalized = array(
            "h",
            (
                max(-32_768, min(full_scale, round(sample * gain)))
                for sample in samples
            ),
        )

        output = BytesIO()
        with wave.open(output, "wb") as wav_file:
            wav_file.setparams(params)
            wav_file.writeframes(normalized.tobytes())
        return output.getvalue()
