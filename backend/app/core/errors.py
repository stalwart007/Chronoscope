"""Domain exceptions.

Each carries an HTTP status and a stable ``code`` so clients can branch on the
code instead of matching message text.
"""

from __future__ import annotations

from typing import Any


class ChronoscopeError(Exception):
    """Base class for all domain errors."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, *, detail: Any = None, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail
        if code:
            self.code = code

    def to_payload(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message, "detail": self.detail}}


class NotFound(ChronoscopeError):
    status_code = 404
    code = "not_found"


class BadRequest(ChronoscopeError):
    status_code = 400
    code = "bad_request"


class UnsupportedMedia(ChronoscopeError):
    status_code = 415
    code = "unsupported_media"


class DependencyUnavailable(ChronoscopeError):
    """A required external binary / service / model is missing."""

    status_code = 503
    code = "dependency_unavailable"


class PipelineError(ChronoscopeError):
    status_code = 500
    code = "pipeline_failed"

    def __init__(self, stage: str, message: str, *, detail: Any = None) -> None:
        super().__init__(f"[{stage}] {message}", detail=detail)
        self.stage = stage


class LLMError(ChronoscopeError):
    status_code = 502
    code = "llm_error"


class SandboxViolation(ChronoscopeError):
    status_code = 400
    code = "sandbox_violation"
