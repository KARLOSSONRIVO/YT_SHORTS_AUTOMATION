from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.exceptions import AppError, ProviderRateLimitError


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        status_code = 500
        if exc.code == "provider_rate_limit":
            status_code = 429
        elif exc.code in {"validation_error", "media_error"}:
            status_code = 422

        return JSONResponse(
            status_code=status_code,
            headers=(
                exc.response_headers
                if isinstance(exc, ProviderRateLimitError)
                else None
            ),
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                }
            },
        )
