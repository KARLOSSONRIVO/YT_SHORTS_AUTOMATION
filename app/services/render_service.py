from dataclasses import dataclass
from pathlib import Path
import random

from app.core.exceptions import IntegrationError
from app.schemas.analysis import ClipCandidate
from app.schemas.media import MediaMetadata
from app.schemas.subtitles import SubtitlePreferences
from app.schemas.transcription import TranscriptResult, TranscriptWord
from app.schemas.workflow import RenderedClipResult
from app.utils.output_paths import output_url, stage_output_dir


@dataclass(slots=True)
class WordBlock:
    start: float
    end: float
    words: list[TranscriptWord]
    lines: list[list[TranscriptWord]]


class RenderService:
    WORD_LEAD_IN_SECONDS = 0.35
    VIDEO_ZOOM_FACTOR = 1.10
    MUSIC_MIN_START_OFFSET_SECONDS = 10.0
    MUSIC_MAX_START_OFFSET_SECONDS = 15.0
    def __init__(self, ffmpeg_client, output_dir: str) -> None:
        self.ffmpeg_client = ffmpeg_client
        self.output_dir = Path(output_dir)

    def render_clips(
        self,
        *,
        media_uri: str,
        job_id: str,
        project_id: str | None,
        project_title: str | None,
        media_metadata: MediaMetadata,
        transcript: TranscriptResult,
        clips: list[ClipCandidate],
        subtitle_prefs: SubtitlePreferences,
    ) -> list[RenderedClipResult]:
        if not self.ffmpeg_client.is_available():
            raise IntegrationError("ffmpeg is required to render final Shorts clips.")

        clips_dir = stage_output_dir(
            output_dir=self.output_dir,
            project_title=project_title,
            project_id=project_id or job_id,
            stage_name="clips",
        )
        clips_dir.mkdir(parents=True, exist_ok=True)
        layout = self._compute_layout(media_metadata)

        rendered: list[RenderedClipResult] = []
        for index, clip in enumerate(clips, start=1):
            clip_basename = self._clip_basename(
                job_id=job_id,
                index=index,
                title_hint=clip.title_hint,
            )
            subtitle_path = clips_dir / f"{clip_basename}.ass"
            output_path = clips_dir / f"{clip_basename}.mp4"
            render_start = max(clip.start - self.WORD_LEAD_IN_SECONDS, 0.0)
            render_end = clip.end
            self._write_ass(
                subtitle_path=subtitle_path,
                transcript=transcript,
                clip_start=render_start,
                clip_end=render_end,
                subtitle_prefs=subtitle_prefs,
                layout=layout,
            )
            self._render_video(
                media_uri=media_uri,
                start=render_start,
                end=render_end,
                subtitle_path=subtitle_path,
                output_path=output_path,
                layout=layout,
            )
            rendered.append(
                RenderedClipResult(
                    clip_index=index,
                    start=round(render_start, 2),
                    end=round(render_end, 2),
                    title_hint=clip.title_hint,
                    score=clip.scores.total_score,
                    video_path=str(output_path.resolve()),
                    video_url=output_url(output_dir=self.output_dir, file_path=output_path),
                    subtitles_path=str(subtitle_path.resolve()),
                )
            )

        return rendered

    def _render_video(
        self,
        *,
        media_uri: str,
        start: float,
        end: float,
        subtitle_path: Path,
        output_path: Path,
        layout: dict,
    ) -> None:
        duration = max(end - start, 0.1)
        escaped_subtitle_path = self._escape_filter_path(subtitle_path)
        video_filter = (
            "scale=1080:1920:force_original_aspect_ratio=decrease,"
            f"scale=iw*{self.VIDEO_ZOOM_FACTOR}:ih*{self.VIDEO_ZOOM_FACTOR},"
            "crop=min(iw\\,1080):ih:(iw-min(iw\\,1080))/2:0,"
            f"pad=1080:1920:(ow-iw)/2:{layout['video_y']}:color=black,"
            f"subtitles='{escaped_subtitle_path}'"
        )

        command = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start),
            "-t",
            str(duration),
            "-i",
            media_uri,
            "-vf",
            video_filter,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        self.ffmpeg_client.run(command)

    def _write_ass(
        self,
        *,
        subtitle_path: Path,
        transcript: TranscriptResult,
        clip_start: float,
        clip_end: float,
        subtitle_prefs: SubtitlePreferences,
        layout: dict,
    ) -> None:
        blocks = self._build_word_blocks(
            transcript=transcript,
            clip_start=clip_start,
            clip_end=clip_end,
            max_chars_per_line=subtitle_prefs.max_chars_per_line,
            max_lines=subtitle_prefs.max_lines,
        )

        lines = [
            "[Script Info]",
            "ScriptType: v4.00+",
            "PlayResX: 1080",
            "PlayResY: 1920",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            self._ass_style_line(subtitle_prefs, layout),
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]

        for block in blocks:
            events = self._build_block_events(
                block=block,
                clip_start=clip_start,
                highlight_color=subtitle_prefs.highlight_color,
            )
            lines.extend(events)

        subtitle_path.write_text("\n".join(lines), encoding="utf-8")

    def _build_word_blocks(
        self,
        *,
        transcript: TranscriptResult,
        clip_start: float,
        clip_end: float,
        max_chars_per_line: int,
        max_lines: int,
    ) -> list[WordBlock]:
        clip_words = self._collect_clip_words(transcript, clip_start, clip_end)
        if not clip_words:
            return []

        blocks: list[WordBlock] = []
        current_words: list[TranscriptWord] = []

        for word in clip_words:
            if not current_words:
                current_words = [word]
                continue

            previous = current_words[-1]
            trial_words = current_words + [word]
            gap = word.start - previous.end

            if gap > 0.8 or not self._fits_block(trial_words, max_chars_per_line, max_lines):
                blocks.append(self._make_block(current_words, max_chars_per_line))
                current_words = [word]
                continue

            current_words.append(word)

        if current_words:
            blocks.append(self._make_block(current_words, max_chars_per_line))

        return blocks

    def _collect_clip_words(
        self,
        transcript: TranscriptResult,
        clip_start: float,
        clip_end: float,
    ) -> list[TranscriptWord]:
        words: list[TranscriptWord] = []
        for segment in transcript.segments:
            if segment.end <= clip_start or segment.start >= clip_end:
                continue

            if segment.words:
                for word in segment.words:
                    if word.end <= clip_start or word.start >= clip_end:
                        continue
                    words.append(
                        TranscriptWord(
                            start=max(word.start, clip_start),
                            end=min(word.end, clip_end),
                            word=word.word.strip(),
                            probability=word.probability,
                        )
                    )
                continue

            words.append(
                TranscriptWord(
                    start=max(segment.start, clip_start),
                    end=min(segment.end, clip_end),
                    word=segment.text.strip(),
                    probability=None,
                )
            )

        return [word for word in words if word.word]

    def _fits_block(
        self,
        words: list[TranscriptWord],
        max_chars_per_line: int,
        max_lines: int,
    ) -> bool:
        line_count = len(self._arrange_lines(words, max_chars_per_line))
        return line_count <= max_lines

    def _make_block(self, words: list[TranscriptWord], max_chars_per_line: int) -> WordBlock:
        lines = self._arrange_lines(words, max_chars_per_line)
        return WordBlock(
            start=words[0].start,
            end=words[-1].end,
            words=words,
            lines=lines,
        )

    def _arrange_lines(
        self,
        words: list[TranscriptWord],
        max_chars_per_line: int,
    ) -> list[list[TranscriptWord]]:
        lines: list[list[TranscriptWord]] = []
        current_line: list[TranscriptWord] = []
        current_length = 0

        for word in words:
            token = word.word.strip()
            if not token:
                continue

            tentative_length = len(token) if not current_line else current_length + 1 + len(token)
            if current_line and tentative_length > max_chars_per_line:
                lines.append(current_line)
                current_line = [word]
                current_length = len(token)
            else:
                current_line.append(word)
                current_length = tentative_length

        if current_line:
            lines.append(current_line)

        return lines

    def _build_block_events(
        self,
        *,
        block: WordBlock,
        clip_start: float,
        highlight_color: str,
    ) -> list[str]:
        events: list[str] = []
        for index, word in enumerate(block.words):
            next_start = block.words[index + 1].start if index + 1 < len(block.words) else block.end
            event_start = max(word.start, block.start)
            event_end = max(next_start, word.end)
            ass_text = self._build_highlighted_text(
                lines=block.lines,
                active_word=word,
                highlight_color=highlight_color,
            )
            events.append(
                "Dialogue: 0,"
                f"{self._format_ass_timestamp(max(event_start - clip_start, 0.0))},"
                f"{self._format_ass_timestamp(max(event_end - clip_start, 0.0))},"
                f"Default,,0,0,0,,{{\\an2\\q2}}{ass_text}"
            )
        return events

    def _build_highlighted_text(
        self,
        *,
        lines: list[list[TranscriptWord]],
        active_word: TranscriptWord,
        highlight_color: str,
    ) -> str:
        rendered_lines: list[str] = []
        ass_color = self._hex_to_ass_color(highlight_color)
        highlight_tag = f"{{\\1c{ass_color}\\c{ass_color}\\b1}}"

        for line in lines:
            tokens: list[str] = []
            for word in line:
                token = self._escape_ass_text(word.word.strip())
                if (
                    word.start == active_word.start
                    and word.end == active_word.end
                    and word.word == active_word.word
                ):
                    tokens.append(f"{highlight_tag}{token}{{\\rDefault}}")
                else:
                    tokens.append(token)
            rendered_lines.append(" ".join(tokens))

        return r"\N".join(rendered_lines)

    def _clip_basename(
        self,
        *,
        job_id: str,
        index: int,
        title_hint: str | None,
    ) -> str:
        if title_hint:
            sanitized_title = "".join(
                character.lower() if character.isalnum() else "_"
                for character in title_hint.strip()
            )
            sanitized_title = "_".join(part for part in sanitized_title.split("_") if part)
            if sanitized_title:
                return f"{sanitized_title[:48]}_{job_id}"

        return f"clip_{index:02d}_{job_id}"

    def _compute_layout(self, media_metadata: MediaMetadata) -> dict:
        video_stream = next((stream for stream in media_metadata.streams if stream.codec_type == "video"), None)
        source_width = video_stream.width if video_stream and video_stream.width else 1080
        source_height = video_stream.height if video_stream and video_stream.height else 1080

        scale_ratio = min(1080 / source_width, 1920 / source_height)
        scaled_height = int(source_height * scale_ratio)

        # Keep the video truly centered vertically in the Shorts canvas.
        video_y = max((1920 - scaled_height) // 2, 0)

        # For bottom-center ASS alignment, MarginV is measured from the bottom.
        # This places the caption slightly above the lower edge of the visible video.
        subtitle_margin_bottom = max(1920 - (video_y + scaled_height) - 10, 10)

        return {
            "video_y": video_y,
            "subtitle_margin_bottom": subtitle_margin_bottom,
        }

    def _ass_style_line(self, prefs: SubtitlePreferences, layout: dict) -> str:
        alignment = 2
        if prefs.position == "top_center":
            alignment = 8

        return (
            "Style: Default,"
            f"{prefs.font_family},"
            f"{max(int(prefs.font_size * 0.72), 30)},"
            f"{self._hex_to_ass_color(prefs.fill_color)},"
            f"{self._hex_to_ass_color(prefs.highlight_color)},"
            f"{self._hex_to_ass_color(prefs.stroke_color)},"
            "&H64000000,"
            "0,0,0,0,100,100,0,0,1,2,0,"
            f"{alignment},80,80,{layout['subtitle_margin_bottom']},1"
        )

    def _format_ass_timestamp(self, seconds: float) -> str:
        total_centiseconds = int(round(seconds * 100))
        hours = total_centiseconds // 360000
        minutes = (total_centiseconds % 360000) // 6000
        secs = (total_centiseconds % 6000) // 100
        centis = total_centiseconds % 100
        return f"{hours}:{minutes:02}:{secs:02}.{centis:02}"

    def _escape_filter_path(self, path: Path) -> str:
        escaped = path.resolve().as_posix().replace("\\", "/")
        escaped = escaped.replace(":", r"\:")
        escaped = escaped.replace("'", r"\'")
        return escaped

    def _escape_ass_text(self, text: str) -> str:
        escaped = text.replace("\\", r"\\")
        escaped = escaped.replace("{", r"\{")
        escaped = escaped.replace("}", r"\}")
        return escaped

    def _hex_to_ass_color(self, value: str) -> str:
        cleaned = value.strip().lstrip("#")
        if len(cleaned) != 6:
            return "&H00FFFFFF"
        rr, gg, bb = cleaned[0:2], cleaned[2:4], cleaned[4:6]
        return f"&H00{bb}{gg}{rr}"

    def mix_audio_with_music(
        self,
        *,
        narration_path: Path,
        music_path: Path,
        output_path: Path,
        ducking: bool = True,
        music_volume: float = 0.15,
    ) -> None:
        if not music_path.exists():
            raise IntegrationError("Background music track was not found.")

        music_volume = min(max(music_volume, 0.0), 1.0)
        narration_duration = self._audio_duration(narration_path)
        if narration_duration is None:
            raise IntegrationError("Narration duration could not be determined for music mixing.")
        music_start_offset = round(
            random.uniform(self.MUSIC_MIN_START_OFFSET_SECONDS, self.MUSIC_MAX_START_OFFSET_SECONDS),
            2,
        )
        narration_filter = "[0:a]aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo[narr]"
        music_filter = (
            f"[1:a]aresample=44100,"
            "aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"atrim=start={music_start_offset}:duration={narration_duration},"
            f"asetpts=PTS-STARTPTS,volume={music_volume}[music]"
        )

        if ducking:
            filter_complex = (
                f"{narration_filter};"
                f"{music_filter};"
                "[music][narr]sidechaincompress=threshold=0.02:ratio=6:attack=20:release=250[ducked];"
                "[narr][ducked]amix=inputs=2:duration=first:dropout_transition=0,"
                "alimiter=limit=0.95[aout]"
            )
        else:
            filter_complex = (
                f"{narration_filter};"
                f"{music_filter};"
                "[narr][music]amix=inputs=2:duration=first:dropout_transition=0,"
                "alimiter=limit=0.95[aout]"
            )

        self.ffmpeg_client.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(narration_path),
                "-stream_loop",
                "-1",
                "-i",
                str(music_path),
                "-filter_complex",
                filter_complex,
                "-map",
                "[aout]",
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ]
        )

    def _audio_duration(self, audio_path: Path) -> float | None:
        try:
            import wave

            with wave.open(str(audio_path), "rb") as audio_file:
                frame_count = audio_file.getnframes()
                frame_rate = audio_file.getframerate()
                if frame_rate <= 0:
                    return None
                return round(frame_count / float(frame_rate), 2)
        except (FileNotFoundError, wave.Error):
            return None
