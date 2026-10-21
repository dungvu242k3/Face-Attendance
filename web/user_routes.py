"""Employee-facing recognition and attendance routes."""

from __future__ import annotations

import secrets
from typing import Any

from flask import Blueprint, current_app, jsonify, render_template, request, session

from modules.attendance_service_constants import VALID_CHECK_TYPES
from modules.image_utils import InvalidImage, decode_data_url
from modules.recognition_service import FaceRecognitionResult
from web.rate_limit import limiter


user_bp = Blueprint("user", __name__)
USER_SESSION_NONCE_KEY = "user_session_nonce"


def _services() -> Any:
    return current_app.extensions["services"]


def _user_session_token() -> str:
    token = session.get(USER_SESSION_NONCE_KEY)
    if not token:
        token = secrets.token_urlsafe(32)
        session[USER_SESSION_NONCE_KEY] = token
    return token


@user_bp.route("/")
def index():
    return render_template("index.html", api_token=_user_session_token())


@user_bp.route("/api/recognize", methods=["POST"])
@limiter.limit("20 per minute", methods=["POST"])
def recognize():
    services = _services()
    expected_token = session.get(USER_SESSION_NONCE_KEY)
    supplied_token = request.headers.get("X-User-Session-Token")
    if (
        not expected_token
        or not supplied_token
        or not secrets.compare_digest(
            expected_token,
            supplied_token,
        )
    ):
        return jsonify(
            {"status": "error", "message": "Phiên truy cập không hợp lệ"}
        ), 401

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"status": "error", "message": "Payload JSON không hợp lệ"}), 400
    image_value = payload.get("image")
    check_type = payload.get("type", "checkin")

    try:
        if check_type not in VALID_CHECK_TYPES:
            raise ValueError("Loại chấm công không hợp lệ")
        frame = decode_data_url(
            image_value,
            services.settings.max_image_bytes,
            services.settings.max_image_pixels,
            services.settings.max_image_dimension,
        )
        result = services.recognition_service.recognize_frame(frame)
        if result.status != "ok":
            return jsonify(_result_payload(result))

        attendance = services.attendance_service.record(
            employee_id=result.employee_id,
            person_name=result.name,
            check_type=check_type,
            image_path=None,
        )
        action = "đã được ghi nhận trước đó" if attendance.duplicate else "thành công"
        return jsonify(
            {
                "status": "duplicate" if attendance.duplicate else "ok",
                "message": (
                    f"{check_type} {action} - "
                    f"{attendance.check_time.strftime('%H:%M:%S')}"
                ),
                "name": result.name,
                "attendance_status": attendance.status,
            }
        )
    except (InvalidImage, ValueError) as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    except Exception:
        current_app.logger.exception("Recognition request failed")
        return jsonify({"status": "error", "message": "Không thể xử lý chấm công"}), 500


def _result_payload(result: FaceRecognitionResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": result.status,
        "message": result.message,
        "name": result.name,
    }
    return payload
