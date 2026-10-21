"""Flask application factories."""

from __future__ import annotations

import atexit
from datetime import timedelta

from flask import Flask
from flask_cors import CORS

from config import Settings
from modules.service_container import Services, build_services
from web.admin_routes import admin_bp
from web.auth_routes import auth_bp
from web.rate_limit import register_rate_limiter
from web.security import register_security
from web.user_routes import user_bp


def _configure_app(
    app: Flask,
    settings: Settings,
    services: Services,
    session_cookie_name: str,
) -> Flask:
    app.secret_key = settings.secret_key
    app.config.update(
        MAX_CONTENT_LENGTH=settings.max_request_bytes,
        SESSION_COOKIE_NAME=session_cookie_name,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=settings.cookie_secure,
        PERMANENT_SESSION_LIFETIME=timedelta(seconds=settings.session_timeout_seconds),
        SESSION_REFRESH_EACH_REQUEST=False,
        DEBUG=settings.debug,
    )
    app.extensions["settings"] = settings
    app.extensions["services"] = services
    register_rate_limiter(app, settings.rate_limit_storage_uri)
    atexit.register(services.database.close)
    return app


def create_admin_app(
    settings: Settings | None = None,
    services: Services | None = None,
) -> Flask:
    settings = settings or Settings.from_env()
    services = services or build_services(settings, with_liveness=False)
    app = Flask(__name__, static_folder="../static", template_folder="../templates")
    _configure_app(app, settings, services, session_cookie_name="admin_session")
    register_security(app)
    app.register_blueprint(auth_bp)
    app.register_blueprint(admin_bp)
    return app


def create_user_app(
    settings: Settings | None = None,
    services: Services | None = None,
) -> Flask:
    settings = settings or Settings.from_env()
    services = services or build_services(settings, with_liveness=True)
    app = Flask(__name__, static_folder="../static", template_folder="../templates")
    _configure_app(app, settings, services, session_cookie_name="user_session")
    CORS(
        app,
        resources={
            r"/api/*": {
                "origins": list(settings.allowed_origins),
                "methods": ["POST", "OPTIONS"],
                "allow_headers": ["Content-Type", "X-User-Session-Token"],
                "supports_credentials": True,
            }
        },
    )
    register_security(app)
    app.register_blueprint(user_bp)
    return app
