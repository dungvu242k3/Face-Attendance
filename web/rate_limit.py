"""Shared request rate limiting for the web applications."""

from __future__ import annotations

from typing import Any

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address


limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[],
    headers_enabled=True,
    swallow_errors=False,
    in_memory_fallback_enabled=False,
)


def register_rate_limiter(app: Any, storage_uri: str) -> None:
    """Attach the shared limiter to an app.

    Production settings must provide a shared backend such as Redis. The
    in-memory backend is intentionally reserved for local development.
    """

    app.config.update(
        RATELIMIT_STORAGE_URI=storage_uri,
        RATELIMIT_HEADERS_ENABLED=True,
    )
    limiter.init_app(app)
