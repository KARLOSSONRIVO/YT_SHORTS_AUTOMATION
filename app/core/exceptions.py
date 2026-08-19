from collections.abc import Mapping


class AppError(Exception):
    def __init__(self, message: str, *, code: str = "app_error") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class ValidationError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="validation_error")


class MediaError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="media_error")


class IntegrationError(AppError):
    def __init__(self, message: str, *, code: str = "integration_error") -> None:
        super().__init__(message, code=code)


class ProviderRateLimitError(IntegrationError):
    SAFE_RESPONSE_HEADERS = {
        "retry-after",
        "x-ratelimit-limit-requests",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-reset-requests",
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset-tokens",
    }

    def __init__(
        self,
        message: str,
        *,
        response_headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message, code="provider_rate_limit")
        self.response_headers = {
            key.lower(): str(value)
            for key, value in (response_headers or {}).items()
            if key.lower() in self.SAFE_RESPONSE_HEADERS
        }


class ContentSafetyError(IntegrationError):
    """Raised when an image provider rejects a prompt through safety filtering."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="provider_content_safety")


class PaymentRequiredError(IntegrationError):
    """Raised when a remote API returns HTTP 402 (Payment Required).

    Callers can catch this specifically to fall back to a local model
    while still letting other integration errors (401, 500, etc.) propagate.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
