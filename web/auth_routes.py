"""Admin login and logout routes."""

from __future__ import annotations

import hmac
from urllib.parse import urlparse

from flask import (
    Blueprint,
    current_app,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash

from web.rate_limit import limiter
from web.security import SESSION_KEY, csrf_token, validate_csrf


auth_bp = Blueprint("auth", __name__, url_prefix="/admin")


def _safe_next_url(value: str | None) -> str:
    if not value:
        return url_for("admin.dashboard")
    parsed = urlparse(value)
    if parsed.scheme or parsed.netloc or not value.startswith("/"):
        return url_for("admin.dashboard")
    return value


@auth_bp.route("/login", methods=["GET", "POST"])
@limiter.limit("5 per minute", methods=["POST"])
def login():
    settings = current_app.extensions["settings"]
    next_url = request.args.get("next") or request.form.get("next")

    if request.method == "POST":
        if not validate_csrf():
            return render_template(
                "login.html", error="CSRF token không hợp lệ", next_url=next_url
            ), 400
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        log_username = username[:100].replace("\r", "\\r").replace("\n", "\\n")
        if not settings.admin_password and not settings.admin_password_hash:
            return render_template(
                "login.html",
                error="Chưa cấu hình thông tin đăng nhập quản trị",
                next_url=next_url,
            ), 503
        if settings.admin_password_hash:
            try:
                password_valid = check_password_hash(
                    settings.admin_password_hash,
                    password,
                )
            except ValueError:
                password_valid = False
        else:
            password_valid = hmac.compare_digest(password, settings.admin_password)

        if hmac.compare_digest(username, settings.admin_username) and password_valid:
            session.clear()
            session.permanent = True
            session[SESSION_KEY] = True
            session["admin_username"] = settings.admin_username
            csrf_token()
            current_app.logger.info(
                "Admin login succeeded for username=%s from ip=%s",
                settings.admin_username,
                request.remote_addr,
            )
            return redirect(_safe_next_url(next_url))
        current_app.logger.warning(
            "Admin login failed for username=%s from ip=%s",
            log_username,
            request.remote_addr,
        )
        return render_template(
            "login.html",
            error="Tên đăng nhập hoặc mật khẩu không đúng",
            next_url=next_url,
        ), 401

    return render_template("login.html", error=None, next_url=next_url)


@auth_bp.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("auth.login"))
