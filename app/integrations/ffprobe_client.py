import json
import shutil
import subprocess
from pathlib import Path

from app.core.exceptions import IntegrationError, MediaError


class FFprobeClient:
    def is_available(self) -> bool:
        return shutil.which("ffprobe") is not None

    def probe(self, media_uri: str) -> dict:
        if not self.is_available():
            raise IntegrationError("ffprobe is not installed or not available on PATH.")

        path = Path(media_uri)
        if not path.exists():
            raise MediaError(f"Media file does not exist: {media_uri}")

        command = [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise IntegrationError(result.stderr.strip() or "ffprobe failed.")
        return json.loads(result.stdout)
