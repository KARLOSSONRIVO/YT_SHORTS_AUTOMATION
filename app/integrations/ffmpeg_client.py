import shutil
import subprocess

from app.core.exceptions import IntegrationError


class FFmpegClient:
    def is_available(self) -> bool:
        return shutil.which("ffmpeg") is not None

    def run(self, command: list[str]) -> None:
        if not self.is_available():
            raise IntegrationError("ffmpeg is not installed or not available on PATH.")

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise IntegrationError(result.stderr.strip() or "ffmpeg command failed.")
