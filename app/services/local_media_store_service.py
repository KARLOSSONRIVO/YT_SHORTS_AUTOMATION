from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile


class LocalMediaStoreService:
    def __init__(self, upload_dir: str) -> None:
        self.upload_dir = Path(upload_dir)

    async def save_upload(self, upload: UploadFile) -> str:
        self.upload_dir.mkdir(parents=True, exist_ok=True)

        suffix = Path(upload.filename or "").suffix or ".bin"
        safe_name = Path(upload.filename or "upload").stem.replace(" ", "_")
        filename = f"{safe_name}_{uuid4().hex}{suffix}"
        target = self.upload_dir / filename

        with target.open("wb") as file_obj:
            while chunk := await upload.read(1024 * 1024):
                file_obj.write(chunk)

        await upload.close()
        return str(target.resolve())

    def delete_upload(self, media_uri: str) -> None:
        target = Path(media_uri)
        try:
            target.unlink(missing_ok=True)
        except OSError:
            # Best-effort cleanup. Render/transcribe/analyze results are already
            # persisted elsewhere, so a stale temp upload should not fail the request.
            pass
