from pathlib import Path


class StorageClient:
    def resolve_local_media(self, media_uri: str) -> Path:
        return Path(media_uri)
