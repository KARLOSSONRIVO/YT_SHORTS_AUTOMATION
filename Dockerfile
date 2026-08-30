FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY app ./app
COPY pyproject.toml README.md ./

EXPOSE 8000

# Shell form so PY_WORKER_PROCESSES is expanded at start-up, with `exec` so
# uvicorn replaces the shell and still receives SIGTERM on `docker stop`.
# Each worker is a separate process with its own whisper model, so raise the
# count against available RAM, not just CPU count.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${PY_WORKER_PROCESSES:-3}"]
