import logging

from app.core.exceptions import MediaError, ValidationError
from app.core.telemetry import stage_timer
from app.schemas.analysis import AnalyzeRequest, AnalyzeResponse
from app.schemas.subtitles import SubtitleGenerationRequest

logger = logging.getLogger(__name__)


class HighlightPipeline:
    def __init__(
        self,
        metadata_service,
        media_prep_service,
        transcription_service,
        transcript_segmentation_service,
        silence_detection_service,
        hook_scoring_service,
        keyword_scoring_service,
        speech_intensity_service,
        clip_detection_service,
        clip_ranking_service,
        subtitle_formatting_service,
    ) -> None:
        self.metadata_service = metadata_service
        self.media_prep_service = media_prep_service
        self.transcription_service = transcription_service
        self.transcript_segmentation_service = transcript_segmentation_service
        self.silence_detection_service = silence_detection_service
        self.hook_scoring_service = hook_scoring_service
        self.keyword_scoring_service = keyword_scoring_service
        self.speech_intensity_service = speech_intensity_service
        self.clip_detection_service = clip_detection_service
        self.clip_ranking_service = clip_ranking_service
        self.subtitle_formatting_service = subtitle_formatting_service

    async def run(self, req: AnalyzeRequest) -> AnalyzeResponse:
        logger.info("highlight_pipeline_started", extra={"job_id": req.job_id, "stage": "analysis"})
        if req.min_clip_duration >= req.max_clip_duration:
            raise ValidationError("min_clip_duration must be smaller than max_clip_duration.")

        with stage_timer("metadata", job_id=req.job_id):
            media = self.metadata_service.read(req.media_uri)
            if not media.has_audio:
                raise MediaError("Media has no audio stream for clip analysis.")
        with stage_timer("media_prep", job_id=req.job_id):
            prepared_media_uri = self.media_prep_service.prepare(req.media_uri)
        with stage_timer("transcription", job_id=req.job_id):
            transcript = await self.transcription_service.transcribe(prepared_media_uri, req.language)
        with stage_timer("windowing", job_id=req.job_id):
            windows = self.transcript_segmentation_service.build_windows(
                transcript=transcript,
                min_duration=req.min_clip_duration,
                max_duration=req.max_clip_duration,
            )
        with stage_timer("silence_detection", job_id=req.job_id):
            boundaries = self.silence_detection_service.detect(transcript)
        with stage_timer("candidate_generation", job_id=req.job_id):
            candidates = self.clip_detection_service.generate_candidates(
                transcript=transcript,
                windows=windows,
                boundaries=boundaries,
                min_duration=req.min_clip_duration,
                max_duration=req.max_clip_duration,
                target_keywords=req.target_keywords,
                top_k=req.top_k,
            )
        with stage_timer("scoring", job_id=req.job_id):
            scored_candidates = []
            for candidate in candidates:
                hook_score = self.hook_scoring_service.score(candidate.transcript_excerpt)
                keyword_score = self.keyword_scoring_service.score(
                    text=candidate.transcript_excerpt,
                    target_keywords=req.target_keywords,
                )
                intensity_score = self.speech_intensity_service.score(
                    text=candidate.transcript_excerpt,
                    duration=max(candidate.end - candidate.start, 0.1),
                )
                candidate.scores.hook_score = hook_score
                candidate.scores.keyword_score = keyword_score
                candidate.scores.speech_intensity_score = intensity_score
                scored_candidates.append(candidate)
        with stage_timer("ranking", job_id=req.job_id):
            ranked_candidates = self.clip_ranking_service.rank(
                candidates=scored_candidates,
                min_duration=req.min_clip_duration,
                max_duration=req.max_clip_duration,
                top_k=req.top_k,
            )
        with stage_timer("subtitles", job_id=req.job_id):
            subtitles = self.subtitle_formatting_service.build_from_transcript(
                request=SubtitleGenerationRequest(
                    job_id=req.job_id,
                    transcript=transcript,
                    max_chars_per_line=req.subtitle_prefs.max_chars_per_line,
                    max_lines=req.subtitle_prefs.max_lines,
                )
            )

        return AnalyzeResponse(
            job_id=req.job_id,
            media=media,
            transcript=transcript,
            clips=ranked_candidates,
            subtitles=subtitles,
            warnings=[],
        )
