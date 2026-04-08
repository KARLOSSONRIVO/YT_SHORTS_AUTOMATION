from typing import Any

from app.core.exceptions import IntegrationError


class WhisperClient:
    def __init__(self, model_name: str = "base") -> None:
        self.model_name = model_name
        self._model = None

    def is_available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return False
        return True

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise IntegrationError(
                "faster-whisper is not installed. Install the media extras to enable transcription."
            ) from exc

        self._model = WhisperModel(self.model_name)
        return self._model

    def transcribe(self, media_uri: str, language: str | None = None) -> dict:
        model = self._load_model()
        segments, info = model.transcribe(
            media_uri,
            language=language,
            vad_filter=True,
            word_timestamps=True,
        )

        normalized_segments = []
        for segment in segments:
            normalized_segments.append(
                {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": segment.text.strip(),
                    "avg_logprob": getattr(segment, "avg_logprob", None),
                    "words": [
                        {
                            "start": float(word.start),
                            "end": float(word.end),
                            "word": word.word,
                            "probability": getattr(word, "probability", None),
                        }
                        for word in (getattr(segment, "words", None) or [])
                    ],
                }
            )

        return {
            "language": info.language,
            "duration": float(getattr(info, "duration", 0.0) or 0.0),
            "segments": normalized_segments,
        }
