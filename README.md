# YouTube Shorts Python Worker

Internal FastAPI worker for:

- media metadata inspection
- transcription
- highlight detection
- subtitle generation

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
uvicorn app.main:app --reload
```

Or install from `requirements.txt`:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Docker

Build and run everything inside Docker:

```bash
docker compose up --build
```

Put source videos in the local `media/` folder and reference them inside the container as `/media/<filename>`.

The API will be available at:

```text
http://localhost:8000
```

Healthcheck:

```bash
curl http://localhost:8000/internal/health
```

Optional transcription dependencies:

```bash
pip install -e .[media]
```

## Endpoints

- `GET /internal/health`
- `POST /internal/transcribe`
- `POST /internal/analyze-clips`
- `POST /internal/generate-subtitles`

Example analyze request:

```json
{
  "job_id": "job_123",
  "media_uri": "C:/media/source.mp4",
  "language": "en",
  "min_clip_duration": 15,
  "max_clip_duration": 45,
  "top_k": 5,
  "target_keywords": ["productivity", "mistake"]
}
```

If you are calling the worker inside Docker, use a mounted path such as:

```json
{
  "job_id": "job_123",
  "media_uri": "/media/source.mp4"
}
```

You can also use the built-in upload endpoints from the GUI instead of passing a `media_uri` manually:

- `POST /internal/transcribe-upload`
- `POST /internal/analyze-clips-upload`
- `POST /internal/process-video-upload`

Uploaded files are stored in the local `uploads/` folder, which is mounted into the container at `/app/uploads`.
Rendered Shorts clips are written to `outputs/` and served from `/outputs/...` so the GUI can preview them in the browser.

## Notes

- `ffprobe` should be available on the machine for metadata extraction.
- `ffmpeg`/`ffprobe` are system dependencies and are not installed by `pip`.
- `faster-whisper` is optional in this scaffold. If it is not installed, transcription requests will return an integration error until the model dependency is added.
- The clip analysis pipeline is MVP-friendly and explainable: transcript windows, silence-gap boundaries, keyword/hook scoring, and duration-aware ranking.
- The Docker image installs `ffmpeg` and the Python requirements, including `faster-whisper`, so you can avoid setting them up directly on Windows.
