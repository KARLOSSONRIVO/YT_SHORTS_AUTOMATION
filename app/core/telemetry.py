import logging
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def stage_timer(stage: str, *, job_id: str | None = None):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "stage_completed",
            extra={"stage": stage, "job_id": job_id or "-", "elapsed_ms": elapsed_ms},
        )
