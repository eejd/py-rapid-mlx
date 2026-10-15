# SPDX-License-Identifier: Apache-2.0
"""CLI opt-in and server lifespan integration (no work when disabled)."""

import ipaddress
from pathlib import Path

_enabled = False
_directory: str | None = None


def validate_exposure(
    *,
    enabled: bool,
    host: str,
    api_key: str | None,
    inherited_listener: bool = False,
) -> None:
    """Refuse filesystem APIs on a non-loopback listener without auth."""
    if not enabled or api_key:
        return
    if inherited_listener:
        raise ValueError(
            "--enable-local-search with --listen-fd requires API-key "
            "authentication because --host cannot attest the inherited socket; "
            "set --api-key or RAPID_MLX_API_KEY"
        )
    normalized = (host or "").strip().lower()
    if normalized == "localhost":
        return
    try:
        if ipaddress.ip_address(normalized).is_loopback:
            return
    except ValueError:
        pass
    raise ValueError(
        "--enable-local-search on a non-loopback --host requires API-key "
        "authentication; set --api-key or RAPID_MLX_API_KEY, or bind to "
        "127.0.0.1/::1"
    )


def configure(enabled: bool, directory: str | None = None) -> None:
    global _enabled, _directory
    _enabled = bool(enabled)
    _directory = directory


def default_directory() -> Path:
    from .first_run import _state_dir

    return _state_dir() / "local-search"


async def startup(app):
    # Remove stale registrations if an embedded server is restarted in process.
    app.router.routes[:] = [
        r
        for r in app.router.routes
        if not getattr(r, "path", "").startswith("/v1/local-search/")
    ]
    app.openapi_schema = None
    if not _enabled:
        return
    from .local_search.service import LocalSearch
    from .routes.local_search import router

    app.state.local_search = LocalSearch(
        Path(_directory).expanduser() if _directory else default_directory()
    )
    app.include_router(router)
    app.state.local_search.start()


async def shutdown(app):
    service = getattr(app.state, "local_search", None)
    if service is not None:
        await service.close()
        del app.state.local_search
