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

## AI providers

Story scripts use Groq. Scene images use Cloudflare Workers AI with
`@cf/black-forest-labs/flux-2-klein-9b`. Narration speech continues to
use Gemini.

Set these values in `.env`:

```text
GROQ_API_KEY=your-groq-api-key
GROQ_MODEL=qwen/qwen3.6-27b
GROQ_FALLBACK_MODEL=llama-3.1-8b-instant
CLOUDFLARE_ACCOUNT_ID=your-cloudflare-account-id
CLOUDFLARE_API_TOKEN=your-cloudflare-workers-ai-token
CLOUDFLARE_IMAGE_MODEL=@cf/black-forest-labs/flux-2-klein-9b
PY_WORKER_CLOUDFLARE_IMAGE_WIDTH=768
PY_WORKER_CLOUDFLARE_IMAGE_HEIGHT=1024
PY_WORKER_CLOUDFLARE_IMAGE_NUM_STEPS=4
PY_WORKER_CLOUDFLARE_IMAGE_TIMEOUT_SECONDS=300
GEMINI_API_KEY=your-gemini-api-key
```

The Cloudflare API token needs `Workers AI - Read` and `Workers AI - Edit`
permissions. FLUX.2 Klein 9B scene images are requested sequentially at 768×1024
and then normalized to the worker's 1080×1920 output frame.

FLUX.2 Klein 9B requests use multipart form data and a fixed four-step process.
The worker keeps the higher-quality `@cf/black-forest-labs/flux-2-dev` model
available for paid usage and the legacy JSON request path for other configured
Cloudflare image models.

`GROQ_MODEL` defaults to `qwen/qwen3.6-27b`. When that model returns HTTP 429,
the worker retries the same request once with `GROQ_FALLBACK_MODEL`, which
defaults to `llama-3.1-8b-instant`. Other errors do not trigger the fallback,
and script generation never falls back to Gemini.

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
