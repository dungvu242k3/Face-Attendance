"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


def _get_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)


def _resolve_path(value: str | None, default: Path) -> Path:
    path = Path(value) if value else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _get_class_ids() -> tuple[int, ...] | None:
    raw_value = os.getenv("YOLO_FACE_CLASS_IDS", "").strip()
    if not raw_value:
        return None
    return tuple(int(value.strip()) for value in raw_value.split(",") if value.strip())


def _get_sha256(name: str) -> str | None:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return None
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} phải là SHA-256 dạng hexadecimal 64 ký tự")
    return value


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings shared by the web applications and CLIs."""

    environment: str
    secret_key: str
    db_name: str
    db_user: str
    db_password: str
    db_host: str
    db_port: int
    db_sslmode: str
    db_sslrootcert: str | None
    db_min_connections: int
    db_max_connections: int

    admin_username: str
    admin_password: str
    admin_password_hash: str

    yolo_model_path: Path
    yolo_model_sha256: str | None
    detector_backend: str
    liveness_model_path: Path
    liveness_model_sha256: str | None
    device: str
    yolo_confidence: float
    yolo_face_class_ids: tuple[int, ...] | None
    recognition_threshold: float
    liveness_real_class: int
    liveness_threshold: float

    max_image_bytes: int
    max_image_pixels: int
    max_image_dimension: int
    duplicate_window_seconds: int
    checkin_deadline_hour: int
    checkin_deadline_minute: int
    checkout_deadline_hour: int
    checkout_deadline_minute: int
    timezone_name: str
    allowed_origins: tuple[str, ...]
    debug: bool
    cookie_secure: bool
    session_timeout_seconds: int
    rate_limit_storage_uri: str

    @property
    def is_production(self) -> bool:
        return self.environment in {"production", "prod"}

    @classmethod
    def from_env(cls) -> "Settings":
        environment = os.getenv("APP_ENV", os.getenv("FLASK_ENV", "development"))
        environment = environment.strip().lower() or "development"
        is_production = environment in {"production", "prod"}

        secret_key = os.getenv("SECRET_KEY", "").strip()
        if not secret_key or secret_key.lower().startswith(("replace-", "change_me")):
            if is_production:
                raise ValueError(
                    "SECRET_KEY bắt buộc phải được cấu hình trong production"
                )
            secret_key = secrets.token_urlsafe(32)
        elif is_production and len(secret_key) < 32:
            raise ValueError("SECRET_KEY production phải có ít nhất 32 ký tự")

        debug = _get_bool("FLASK_DEBUG", False)
        if is_production and debug:
            raise ValueError("Không được bật FLASK_DEBUG trong production")

        cors_value = os.getenv("CORS_ORIGINS", "")
        allowed_origins = tuple(
            origin.strip()
            for origin in (cors_value or "http://localhost:8000").split(",")
            if origin.strip()
        )
        if is_production and (not cors_value.strip() or "*" in allowed_origins):
            raise ValueError("CORS_ORIGINS phải được cấu hình rõ ràng trong production")

        cookie_secure = _get_bool("SESSION_COOKIE_SECURE", is_production)
        if is_production and not cookie_secure:
            raise ValueError("SESSION_COOKIE_SECURE phải là true trong production")

        rate_limit_storage_uri = os.getenv("RATELIMIT_STORAGE_URI", "").strip()
        if not rate_limit_storage_uri:
            rate_limit_storage_uri = "" if is_production else "memory://"
        if is_production and rate_limit_storage_uri.startswith("memory://"):
            raise ValueError(
                "Production phải dùng storage rate-limit dùng chung, ví dụ Redis"
            )

        db_sslmode = (
            os.getenv("DB_SSLMODE", "require" if is_production else "prefer")
            .strip()
            .lower()
        )
        if db_sslmode not in {
            "disable",
            "allow",
            "prefer",
            "require",
            "verify-ca",
            "verify-full",
        }:
            raise ValueError("DB_SSLMODE không hợp lệ")
        if is_production and db_sslmode not in {"require", "verify-ca", "verify-full"}:
            raise ValueError(
                "Production phải dùng DB_SSLMODE require/verify-ca/verify-full"
            )

        detector_backend = os.getenv("DETECTOR_BACKEND", "haar").strip().lower()
        yolo_model_sha256 = _get_sha256("YOLO_MODEL_SHA256")
        liveness_model_sha256 = _get_sha256("LIVENESS_MODEL_SHA256")
        if is_production and not liveness_model_sha256:
            raise ValueError(
                "LIVENESS_MODEL_SHA256 bắt buộc phải được cấu hình trong production"
            )
        if is_production and detector_backend == "yolo" and not yolo_model_sha256:
            raise ValueError(
                "YOLO_MODEL_SHA256 bắt buộc khi dùng YOLO trong production"
            )

        admin_password_hash = os.getenv("ADMIN_PASSWORD_HASH", "").strip()
        if is_production and (
            not admin_password_hash
            or admin_password_hash.lower().startswith("replace-")
        ):
            raise ValueError(
                "ADMIN_PASSWORD_HASH bắt buộc phải được cấu hình trong production"
            )

        liveness_threshold = _get_float("LIVENESS_THRESHOLD", 0.5)
        if not 0.0 < liveness_threshold <= 1.0:
            raise ValueError("LIVENESS_THRESHOLD phải nằm trong khoảng (0, 1]")
        liveness_real_class = _get_int("LIVENESS_REAL_CLASS", 1)
        if liveness_real_class not in {0, 1}:
            raise ValueError("LIVENESS_REAL_CLASS phải là 0 hoặc 1")

        return cls(
            environment=environment,
            secret_key=secret_key,
            db_name=os.getenv("DB_NAME", "face_db"),
            db_user=os.getenv("DB_USER", "postgres"),
            db_password=os.getenv("DB_PASSWORD", ""),
            db_host=os.getenv("DB_HOST", "localhost"),
            db_port=_get_int("DB_PORT", 5432),
            db_sslmode=db_sslmode,
            db_sslrootcert=os.getenv("DB_SSLROOTCERT") or None,
            db_min_connections=max(1, _get_int("DB_MIN_CONNECTIONS", 1)),
            db_max_connections=max(1, _get_int("DB_MAX_CONNECTIONS", 10)),
            admin_username=os.getenv("ADMIN_USERNAME", "admin"),
            admin_password=os.getenv("ADMIN_PASSWORD", ""),
            admin_password_hash=admin_password_hash,
            yolo_model_path=_resolve_path(
                os.getenv("YOLO_MODEL_PATH"),
                PROJECT_ROOT / "models" / "face_detector.pt",
            ),
            yolo_model_sha256=yolo_model_sha256,
            detector_backend=detector_backend,
            liveness_model_path=_resolve_path(
                os.getenv("LIVENESS_MODEL_PATH"),
                PROJECT_ROOT / "models" / "best_model.pth",
            ),
            liveness_model_sha256=liveness_model_sha256,
            device=os.getenv("DEVICE", "auto").strip().lower(),
            yolo_confidence=_get_float("YOLO_CONFIDENCE", 0.5),
            yolo_face_class_ids=_get_class_ids(),
            recognition_threshold=_get_float("RECOGNITION_THRESHOLD", 0.6),
            liveness_real_class=liveness_real_class,
            liveness_threshold=liveness_threshold,
            max_image_bytes=_get_int("MAX_IMAGE_BYTES", 5 * 1024 * 1024),
            max_image_pixels=max(1, _get_int("MAX_IMAGE_PIXELS", 4_000_000)),
            max_image_dimension=max(1, _get_int("MAX_IMAGE_DIMENSION", 4096)),
            duplicate_window_seconds=_get_int("DUPLICATE_WINDOW_SECONDS", 60),
            checkin_deadline_hour=_get_int("CHECKIN_DEADLINE_HOUR", 9),
            checkin_deadline_minute=_get_int("CHECKIN_DEADLINE_MINUTE", 0),
            checkout_deadline_hour=_get_int("CHECKOUT_DEADLINE_HOUR", 18),
            checkout_deadline_minute=_get_int("CHECKOUT_DEADLINE_MINUTE", 0),
            timezone_name=os.getenv("APP_TIMEZONE", "Asia/Ho_Chi_Minh"),
            allowed_origins=allowed_origins,
            debug=debug,
            cookie_secure=cookie_secure,
            session_timeout_seconds=max(60, _get_int("SESSION_TIMEOUT_SECONDS", 1800)),
            rate_limit_storage_uri=rate_limit_storage_uri,
        )

    @property
    def max_request_bytes(self) -> int:
        """Allow a small amount of multipart/base64 request overhead."""

        return int(self.max_image_bytes * 1.4)
