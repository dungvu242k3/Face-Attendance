"""Minimal session authentication and CSRF protection for admin routes."""

from __future__ import annotations

import secrets
from typing import Any

from flask import jsonify, redirect, request, session, url_for


SESSION_KEY = "admin_authenticated"
CSRF_SESSION_KEY = "csrf_token"


def csrf_token() -> str:
    token = session.get(CSRF_SESSION_KEY)
    if not token:
        token = secrets.token_urlsafe(32)
        session[CSRF_SESSION_KEY] = token
    return token


def validate_csrf() -> bool:
    expected = session.get(CSRF_SESSION_KEY)
    supplied = request.headers.get("X-CSRF-Token") or request.form.get("csrf_token")
    return bool(expected and supplied and secrets.compare_digest(expected, supplied))


def is_admin_authenticated() -> bool:
    return bool(session.get(SESSION_KEY))


def register_security(app: Any) -> None:
    app.context_processor(lambda: {"csrf_token": csrf_token})
    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=app.config.get("SESSION_COOKIE_SECURE", False),
    )

    @app.after_request
    def add_security_headers(response: Any) -> Any:
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Permissions-Policy",
            "camera=(self), microphone=(), geolocation=()",
        )
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; base-uri 'self'; form-action 'self'; "
            "frame-ancestors 'none'; object-src 'none'; img-src 'self' data:; "
            "media-src 'self' blob:; connect-src 'self'; "
            "script-src 'self' https://cdn.tailwindcss.com; "
            "style-src 'self' 'unsafe-inline'",
        )
        if app.config.get("SESSION_COOKIE_SECURE"):
            response.headers.setdefault(
                "Strict-Transport-Security",
                "max-age=31536000; includeSubDomains",
            )
        if request.path.startswith("/admin") or request.path == "/":
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.before_request
    def protect_admin_requests() -> Any:
        if not request.path.startswith("/admin"):
            return None
        if request.path == "/admin/login":
            return None
        if not is_admin_authenticated():
            if request.accept_mimetypes.best == "application/json":
                return jsonify({"status": "error", "message": "Chưa đăng nhập"}), 401
            return redirect(url_for("auth.login", next=request.path))
        if request.method in {"POST", "PUT", "PATCH", "DELETE"} and not validate_csrf():
            return jsonify(
                {"status": "error", "message": "CSRF token không hợp lệ"}
            ), 400
        return None
