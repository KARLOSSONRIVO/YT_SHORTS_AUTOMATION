from fastapi import APIRouter

from app.api.routes.analyze import router as analyze_router
from app.api.routes.health import router as health_router
from app.api.routes.render import router as render_router
from app.api.routes.subtitles import router as subtitles_router
from app.api.routes.transcribe import router as transcribe_router
from app.api.routes.workflow import router as workflow_router

api_router = APIRouter()
api_router.include_router(health_router, tags=["health"])
api_router.include_router(transcribe_router, tags=["transcription"])
api_router.include_router(analyze_router, tags=["analysis"])
api_router.include_router(render_router, tags=["render"])
api_router.include_router(subtitles_router, tags=["subtitles"])
api_router.include_router(workflow_router, tags=["workflow"])
