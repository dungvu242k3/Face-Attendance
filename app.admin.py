import base64
import os
from io import BytesIO

import cv2
import numpy as np
import psycopg2
from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from modules.database import FaceDatabase
from modules.detector import FaceDetector

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "change_me")

app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024


@app.errorhandler(413)
def too_large(e):
    return jsonify({"status": "error", "message": "Ảnh quá lớn (>5MB). Vui lòng chọn ảnh khác."}), 413

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME", "face_db"),
    user=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASSWORD", "sat24042003"),
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", 5432)),
)
cur = conn.cursor()

detector = FaceDetector(device="cuda" if not os.getenv("USE_MOCK_DETECTOR", "true") == "true" else "cpu")
face_db = FaceDatabase()


@app.route("/")
def index():
    return redirect(url_for("dashboard"))


@app.route("/admin/dashboard")
def dashboard():
    try:
        cur.execute("SELECT COUNT(*) FROM employees")
        total_employees = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM attendance_logs_512 WHERE DATE(check_time) = CURRENT_DATE")
        today_logs = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM attendance_logs_512 WHERE DATE(check_time) = CURRENT_DATE AND status='late'")
        late_today = cur.fetchone()[0]

        cur.execute(
            """
            SELECT person_name, check_time, status, check_type
            FROM attendance_logs_512
            ORDER BY check_time DESC
            LIMIT 5
            """
        )
        recent = cur.fetchall()

        return render_template(
            "dashboard.html",
            total_employees=total_employees,
            today_logs=today_logs,
            late_today=late_today,
            recent=recent,
        )
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500


@app.route("/admin/employees")
def employees():
    search_name = request.args.get("name", "")
    position = request.args.get("position", "")

    query = "SELECT id, name, email, phone, position, created_at FROM employees WHERE name LIKE %s"
    params = [f"%{search_name}%"]

    if position:
        query += " AND position = %s"
        params.append(position)

    query += " ORDER BY created_at DESC"
    cur.execute(query, params)
    rows = cur.fetchall()

    cur.execute("SELECT DISTINCT position FROM employees")
    positions = [r[0] for r in cur.fetchall()]

    return render_template(
        "employees.html",
        employees=rows,
        positions=positions,
        search_name=search_name,
        filter_position=position,
    )


@app.route("/admin/add_employee", methods=["GET", "POST"])
def add_employee():
    if request.method == "GET":
        return render_template("add_employee.html")

    try:
        name = (request.form.get("name") or "").strip()
        email = request.form.get("email")
        phone = request.form.get("phone")
        position = request.form.get("position")

        if not name:
            return jsonify({"status": "error", "message": "Tên là bắt buộc"}), 400

        MAX_BYTES = app.config['MAX_CONTENT_LENGTH']

        image_bytes = None

        file = request.files.get("face_image")
        if file and getattr(file, "filename", ""):
            file_bytes = file.read()
            if len(file_bytes) > MAX_BYTES:
                return jsonify({"status": "error", "message": f"File ảnh vượt quá {MAX_BYTES//(1024*1024)}MB!"}), 413
            image_bytes = file_bytes

        else:
            img_data = request.form.get("face_image_webcam")
            if img_data and img_data.startswith("data:image"):
                header, encoded = img_data.split(",", 1)
                try:
                    image_bytes = base64.b64decode(encoded)
                except Exception:
                    return jsonify({"status": "error", "message": "Dữ liệu ảnh webcam không hợp lệ"}), 400

                if len(image_bytes) > MAX_BYTES:
                    return jsonify({"status": "error", "message": f"Ảnh webcam vượt quá {MAX_BYTES//(1024*1024)}MB!"}), 413

        if not image_bytes:
            return jsonify({"status": "error", "message": "Chưa có ảnh"}), 400

        file_np = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(file_np, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"status": "error", "message": "Ảnh không hợp lệ"}), 400

        boxes = detector.detect_faces(img)
        if not boxes:
            return jsonify({"status": "error", "message": "Không phát hiện khuôn mặt"}), 400

        x1, y1, x2, y2 = boxes[0]
        x1, y1 = max(int(x1), 0), max(int(y1), 0)
        x2, y2 = max(int(x2), x1 + 1), max(int(y2), y1 + 1)
        face_img = img[y1:y2, x1:x2]

        embedding = detector.get_embeddings(face_img)
        if isinstance(embedding, np.ndarray):
            embedding_np = embedding.astype(np.float32)
        else:
            embedding_np = embedding.cpu().numpy().astype(np.float32)

        embedding_str = "[" + ",".join(map(str, embedding_np)) + "]"

        cur.execute(
            "INSERT INTO employees (name, email, phone, position) VALUES (%s, %s, %s, %s) RETURNING id",
            (name, email, phone, position),
        )
        employee_id = cur.fetchone()[0]

        cur.execute(
            """
            INSERT INTO face_embeddings_512 (employee_id, person_name, embedding, filename)
            VALUES (%s, %s, %s, %s)
            """,
            (employee_id, name, embedding_str, None),
        )

        conn.commit()

        return jsonify(
            {
                "status": "ok",
                "message": "Thêm nhân viên thành công",
                "employee_id": employee_id,
            }
        )
    except Exception as e:
        conn.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500




@app.route("/admin/edit_employee/<int:emp_id>", methods=["GET", "POST"])
def edit_employee(emp_id):
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        phone = request.form.get("phone")
        position = request.form.get("position")
        try:
            cur.execute(
                """
                UPDATE employees
                SET name=%s, email=%s, phone=%s, position=%s
                WHERE id=%s
                """,
                (name, email, phone, position, emp_id),
            )
            conn.commit()
            return jsonify({"success": True, "message": "Cập nhật thành công"})
        except Exception as e:
            conn.rollback()
            return jsonify({"success": False, "message": str(e)}), 500

    cur.execute("SELECT id, name, email, phone, position, created_at FROM employees WHERE id=%s", (emp_id,))
    employee = cur.fetchone()
    return render_template("edit_employee.html", employee=employee)



@app.route("/admin/delete_employee/<int:emp_id>", methods=["POST"])
def delete_employee(emp_id):
    try:
        cur.execute("DELETE FROM face_embeddings_512 WHERE employee_id = %s", (emp_id,))
        cur.execute("DELETE FROM employees WHERE id = %s RETURNING id", (emp_id,))
        deleted = cur.fetchone()
        if deleted is None:
            conn.rollback()
            return jsonify({"success": False, "message": "Nhân viên không tồn tại"}), 404
        conn.commit()
        return jsonify({"success": True, "message": "Đã xoá nhân viên!"})
    except Exception as e:
        conn.rollback()
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/admin/attendance_history")
def attendance_history():
    check_type = request.args.get("type")
    search_name = request.args.get("name", "")
    try:
        query = """
            SELECT id, employee_id, person_name, check_time, status, image_path, check_type
            FROM attendance_logs_512
            WHERE person_name LIKE %s
        """
        params = [f"%{search_name}%"]
        if check_type in ("checkin", "checkout"):
            query += " AND check_type = %s"
            params.append(check_type)
        query += " ORDER BY check_time DESC LIMIT 200"

        cur.execute(query, params)
        rows = cur.fetchall()
        return render_template("history.html", logs=rows)
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
