"""Shared terminal-state mapping for public serving protocols."""

ENGINE_CANCELLATION_FINISH_REASON = "cancelled"


def public_finish_reason(reason: str | None, error_kind: str | None) -> str | None:
    """Keep a repetition abort inside the protocol's standard length reason."""

    return "length" if error_kind == "repetition" else reason


def is_cancellation_finish_reason(finish_reason: str | None) -> bool:
    """Whether an engine finish reason denotes an inference cancellation."""

    return finish_reason == ENGINE_CANCELLATION_FINISH_REASON


def cancellation_error(
    *,
    type_: str = "server_error",
    code: str = "request_cancelled",
    message: str = "The request was cancelled.",
) -> dict[str, str]:
    """Build the public error payload for a cancellation."""

    return {"type": type_, "code": code, "message": message}
