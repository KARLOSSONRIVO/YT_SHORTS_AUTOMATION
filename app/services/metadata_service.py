from app.schemas.media import MediaMetadata, MediaStream


class MetadataService:
    def __init__(self, ffprobe_client) -> None:
        self.ffprobe_client = ffprobe_client

    def read(self, media_uri: str) -> MediaMetadata:
        payload = self.ffprobe_client.probe(media_uri)
        streams = []
        has_audio = False
        has_video = False

        for stream in payload.get("streams", []):
            codec_type = stream.get("codec_type", "unknown")
            has_audio = has_audio or codec_type == "audio"
            has_video = has_video or codec_type == "video"
            streams.append(
                MediaStream(
                    index=stream.get("index", 0),
                    codec_type=codec_type,
                    codec_name=stream.get("codec_name"),
                    sample_rate=int(stream["sample_rate"]) if stream.get("sample_rate") else None,
                    channels=stream.get("channels"),
                    width=stream.get("width"),
                    height=stream.get("height"),
                )
            )

        fmt = payload.get("format", {})
        return MediaMetadata(
            media_uri=media_uri,
            duration_seconds=float(fmt.get("duration", 0.0)),
            format_name=fmt.get("format_name"),
            size_bytes=int(fmt["size"]) if fmt.get("size") else None,
            has_audio=has_audio,
            has_video=has_video,
            streams=streams,
        )
