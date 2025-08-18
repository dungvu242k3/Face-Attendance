import base64
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from modules.database import FaceDatabase
from modules.detector import FaceDetector
from modules.livenessnet import LivenessNet
from modules.recognition import FaceRecognizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

detector = FaceDetector(device=DEVICE)

liveness_model = LivenessNet(width=32, height=32, depth=3, classes=2)
checkpoint = torch.load("./models/best_model.pth", map_location=DEVICE)
liveness_model.load_state_dict(checkpoint["model_state_dict"])
liveness_model.to(DEVICE)
liveness_model.eval()

db = FaceDatabase(
    dbname="face_db",
    user="postgres",
    password="sat24042003",
    host="localhost",
    port=5432
)
recognizer = FaceRecognizer(db, threshold=0.6)

def preprocess_for_liveness(face_img):
    img = cv2.resize(face_img, (32, 32))
    img = img.astype("float32") / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.tensor(img).unsqueeze(0).to(DEVICE)

def log_attendance(employee_id, name, check_type, image_path=None):
    check_time = datetime.now()
    status = "Present" if check_time.hour < 9 else "Late"
    try:
        cur = db.conn.cursor()
        cur.execute(
            "INSERT INTO attendance_logs_512 (employee_id, person_name, status, check_type, check_time, image_path) VALUES (%s, %s, %s, %s, %s, %s)",
            (employee_id, name, status, check_type, check_time, image_path)
        )
        db.conn.commit()
        cur.close()
    except Exception as e:
        db.conn.rollback()
        print("Error logging attendance:", e)



@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/recognize", methods=["POST"])
def recognize():
    data = request.get_json()
    if "image" not in data:
        return jsonify({"status": "error", "message": "No image provided"}), 400

    check_type = data.get("type", "checkin")
    check_time = datetime.now().strftime("%H:%M")
    img_data = base64.b64decode(data["image"].split(",")[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    boxes = detector.detect_faces(frame)
    if not boxes:
        return jsonify({"status": "no_face", "message": "Không phát hiện khuôn mặt"})

    faces = detector.extract_faces(frame, boxes)
    embeddings = [detector.get_embeddings(face) for face in faces]

    for box, face_img, embedding in zip(boxes, faces, embeddings):
        tensor = preprocess_for_liveness(face_img)
        with torch.no_grad():
            output = liveness_model(tensor)
            pred = torch.argmax(output, dim=1).item()
            is_real = (pred == 1)

        if not is_real:
            return jsonify({"status": "fake", "message": "Phát hiện ảnh giả", "name": "FAKE"})

        employee_id, name, dist, conf = recognizer.recognize(embedding)

        if name == "Unknown":
            return jsonify({"status": "unknown", "message": "Không nhận diện được", "name": "Unknown"})

        log_attendance(employee_id, name, check_type, None)


        return jsonify({"status": "ok", "message": f"{check_type} thành công - {check_time}", "name": name})

    return jsonify({"status": "error", "message": "Không có kết quả"})


@app.route("/history")
def history():
    cur = db.conn.cursor()
    cur.execute("""
        SELECT id, person_name, status, check_time, image_path
        FROM attendance_logs_128
        ORDER BY check_time DESC
    """)
    records = cur.fetchall()
    cur.close()
    return render_template("history.html", records=records)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
