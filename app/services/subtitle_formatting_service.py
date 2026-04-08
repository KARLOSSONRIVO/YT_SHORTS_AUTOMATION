from app.schemas.subtitles import SubtitleGenerationRequest, SubtitleSegment
from app.utils.text import split_caption_lines


class SubtitleFormattingService:
    def build_from_transcript(self, request: SubtitleGenerationRequest) -> list[SubtitleSegment]:
        subtitles: list[SubtitleSegment] = []
        for segment in request.transcript.segments:
            if request.clip_start is not None and segment.end < request.clip_start:
                continue
            if request.clip_end is not None and segment.start > request.clip_end:
                continue

            text = split_caption_lines(
                segment.text,
                max_chars_per_line=request.max_chars_per_line,
                max_lines=request.max_lines,
            )
            subtitles.append(
                SubtitleSegment(
                    start=round(segment.start, 2),
                    end=round(segment.end, 2),
                    text=text,
                )
            )
        return subtitles
