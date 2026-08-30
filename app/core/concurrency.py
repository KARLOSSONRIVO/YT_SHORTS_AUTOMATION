"""Off-loop execution for the blocking, CPU-bound parts of the pipeline.

The FastAPI routes in this service are ``async def``, but the work they call
into is synchronous and long-running: ``subprocess.run`` for ffmpeg and ffprobe,
faster-whisper transcription, and Pillow image work. Awaiting nothing and
calling those directly from a coroutine pins the event loop for the entire
render, which stalls every other request the process is serving - including
``/health``.

``run_blocking`` moves that work onto a worker thread. The GIL is released while
ffmpeg runs as a subprocess and while whisper computes in native code, so
threads give real parallelism here. A capacity limiter bounds how many heavy
jobs one process accepts at a time, so raising the uvicorn worker count scales
throughput without oversubscribing the CPU.
"""

from __future__ import annotations

import functools
from functools import lru_cache
from typing import Any, Callable, TypeVar

import anyio
import anyio.to_thread

from app.core.config import get_settings

T = TypeVar("T")


@lru_cache(maxsize=1)
def get_job_limiter() -> anyio.CapacityLimiter:
    """Per-process cap on concurrently running blocking jobs.

    Built lazily on first use so construction happens inside the running event
    loop, and cached so every caller shares one limiter per worker process.
    """
    return anyio.CapacityLimiter(get_settings().max_concurrent_jobs)


async def run_blocking(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Runs a synchronous callable on a worker thread, bounded by the limiter.

    Exceptions propagate to the caller unchanged, so route-level error handling
    and the registered exception handlers keep working as before.
    """
    call = functools.partial(func, *args, **kwargs)
    return await anyio.to_thread.run_sync(call, limiter=get_job_limiter())
