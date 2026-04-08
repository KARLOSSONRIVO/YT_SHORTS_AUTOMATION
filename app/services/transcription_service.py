from app.schemas.transcription import TranscriptResult, TranscriptSegment, TranscriptWord


class TranscriptionService:
    def __init__(self, whisper_client) -> None:
        self.whisper_client = whisper_client

    async def transcribe(self, media_uri: str, language: str | None) -> TranscriptResult:
        raw = self.whisper_client.transcribe(media_uri, language=language)
        segments = []
        for segment in raw["segments"]:
            segments.append(
                TranscriptSegment(
                    start=segment["start"],
                    end=segment["end"],
                    text=segment["text"],
                    avg_logprob=segment.get("avg_logprob"),
                    words=[
                        TranscriptWord(
                            start=word["start"],
                            end=word["end"],
                            word=word["word"],
                            probability=word.get("probability"),
                        )
                        for word in segment.get("words", [])
                    ],
                )
            )

        return TranscriptResult(
            language=raw["language"],
            duration=float(raw.get("duration") or (segments[-1].end if segments else 0.0)),
            segments=segments,
            full_text=" ".join(segment.text for segment in segments).strip(),
        )
