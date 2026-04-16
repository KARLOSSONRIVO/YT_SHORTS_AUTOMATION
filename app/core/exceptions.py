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
    def __init__(self, message: str) -> None:
        super().__init__(message, code="integration_error")


class PaymentRequiredError(IntegrationError):
    """Raised when a remote API returns HTTP 402 (Payment Required).

    Callers can catch this specifically to fall back to a local model
    while still letting other integration errors (401, 500, etc.) propagate.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
