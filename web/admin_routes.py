"""Admin dashboard routes."""

from __future__ import annotations

from typing import Any

import numpy as np
from flask import (
    Blueprint,
    abort,
    current_app,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)

from modules.image_utils import InvalidImage, decode_data_url, decode_image_bytes


admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


def _services() -> Any:
    return current_app.extensions["services"]


def _clean_optional(value: str | None) -> str | None:
    value = (value or "").strip()
    return value or None


def _read_face_image() -> np.ndarray:
    settings = current_app.extensions["settings"]
    file = request.files.get("face_image")
    if file and file.filename:
        image_bytes = file.read(settings.max_image_bytes + 1)
        return decode_image_bytes(
            image_bytes,
            settings.max_image_bytes,
            settings.max_image_pixels,
            settings.max_image_dimension,
        )

    webcam_value = request.form.get("face_image_webcam")
    return decode_data_url(
        webcam_value,
        settings.max_image_bytes,
        settings.max_image_pixels,
        settings.max_image_dimension,
    )


def _extract_single_embedding(image: np.ndarray) -> np.ndarray:
    services = _services()
    boxes = services.detector.detect_faces(image)
    crops = [
        face
        for box in boxes
        if (face := services.detector.crop_face(image, box)) is not None
    ]
    if not crops:
        raise InvalidImage("Không phát hiện khuôn mặt")
    if len(crops) > 1:
        raise InvalidImage("Ảnh chỉ được chứa một khuôn mặt")
    return services.detector.get_embeddings(crops[0])


@admin_bp.app_errorhandler(413)
def too_large(_: Any):
    return jsonify({"status": "error", "message": "Ảnh quá lớn"}), 413


@admin_bp.route("/")
def index():
    return redirect(url_for("admin.dashboard"))


@admin_bp.route("/dashboard")
def dashboard():
    services = _services()
    start, end = services.attendance_service.day_bounds()
    return render_template(
        "dashboard.html",
        total_employees=services.employees.count(),
        today_logs=services.attendance.count_between(start, end),
        late_today=services.attendance.count_late_between(start, end),
        recent=services.attendance.recent(limit=5),
    )


@admin_bp.route("/employees")
def employees():
    services = _services()
    search_name = request.args.get("name", "").strip()
    position = request.args.get("position", "").strip()
    return render_template(
        "employees.html",
        employees=services.employees.list(name=search_name, position=position),
        positions=services.employees.positions(),
        search_name=search_name,
        filter_position=position,
    )


@admin_bp.route("/add_employee", methods=["GET", "POST"])
def add_employee():
    if request.method == "GET":
        return render_template("add_employee.html")

    services = _services()
    try:
        name = request.form.get("name", "")
        email = _clean_optional(request.form.get("email"))
        phone = _clean_optional(request.form.get("phone"))
        position = _clean_optional(request.form.get("position"))
        image = _read_face_image()
        embedding = _extract_single_embedding(image)
        employee_id = services.employee_service.create_with_face(
            name=name,
            email=email,
            phone=phone,
            position=position,
            embedding=embedding,
        )
        current_app.logger.info(
            "Employee created employee_id=%s from ip=%s",
            employee_id,
            request.remote_addr,
        )
        return jsonify(
            {
                "status": "ok",
                "message": "Thêm nhân viên thành công",
                "employee_id": employee_id,
            }
        )
    except (InvalidImage, ValueError) as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    except Exception:
        current_app.logger.exception("Unable to add employee")
        return jsonify({"status": "error", "message": "Không thể thêm nhân viên"}), 500


@admin_bp.route("/edit_employee/<int:emp_id>", methods=["GET", "POST"])
def edit_employee(emp_id: int):
    services = _services()
    if request.method == "GET":
        employee = services.employees.get(emp_id)
        if employee is None:
            abort(404)
        return render_template("edit_employee.html", employee=employee)

    try:
        updated = services.employee_service.update(
            employee_id=emp_id,
            name=request.form.get("name", ""),
            email=_clean_optional(request.form.get("email")),
            phone=_clean_optional(request.form.get("phone")),
            position=_clean_optional(request.form.get("position")),
        )
        if not updated:
            return jsonify(
                {"success": False, "message": "Nhân viên không tồn tại"}
            ), 404
        current_app.logger.info(
            "Employee updated employee_id=%s from ip=%s",
            emp_id,
            request.remote_addr,
        )
        return jsonify({"success": True, "message": "Cập nhật thành công"})
    except ValueError as exc:
        return jsonify({"success": False, "message": str(exc)}), 400
    except Exception:
        current_app.logger.exception("Unable to update employee %s", emp_id)
        return jsonify(
            {"success": False, "message": "Không thể cập nhật nhân viên"}
        ), 500


@admin_bp.route("/delete_employee/<int:emp_id>", methods=["POST"])
def delete_employee(emp_id: int):
    try:
        deleted = _services().employee_service.delete(emp_id)
        if not deleted:
            return jsonify(
                {"success": False, "message": "Nhân viên không tồn tại"}
            ), 404
        current_app.logger.info(
            "Employee deleted employee_id=%s from ip=%s",
            emp_id,
            request.remote_addr,
        )
        return jsonify({"success": True, "message": "Đã xoá nhân viên"})
    except Exception:
        current_app.logger.exception("Unable to delete employee %s", emp_id)
        return jsonify({"success": False, "message": "Không thể xoá nhân viên"}), 500


@admin_bp.route("/attendance_history")
def attendance_history():
    check_type = request.args.get("type")
    search_name = request.args.get("name", "").strip()
    logs = _services().attendance.search(name=search_name, check_type=check_type)
    return render_template("history.html", logs=logs)
