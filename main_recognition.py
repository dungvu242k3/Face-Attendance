import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from modules.camera import Camera
from modules.database import FaceDatabase
from modules.detector import FaceDetector
from modules.livenessnet import LivenessNet
from modules.recognition import FaceRecognizer


def draw_text_unicode(frame, text, position, font_size=20, color=(0, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype("arial.ttf", font_size)
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def preprocess_for_liveness(face_img):
    img = cv2.resize(face_img, (32, 32))
    img = img.astype("float32") / 255.0
    img = np.transpose(img, (2, 0, 1))  # C x H x W
    return torch.tensor(img).unsqueeze(0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    detector = FaceDetector(device)

    database = FaceDatabase(
        dbname="face_db",
        user="postgres",
        password="sat24042003",
        host="localhost",
        port=5432
    )

    liveness_detector = LivenessNet(width=32, height=32, depth=3, classes=2)
    checkpoint = torch.load("./models/best.pth", map_location=device)
    liveness_detector.load_state_dict(checkpoint["model_state_dict"])
    liveness_detector.to(device)
    liveness_detector.eval()

    recognizer = FaceRecognizer(database, threshold=0.6)
    camera = Camera()

    while True:
        ret, frame = camera.read_frame()
        if not ret:
            break

        boxes = detector.detect_faces(frame)
        faces = detector.extract_faces(frame, boxes) if boxes else []

        unknown_faces = []

        for box, face_img in zip(boxes, faces):
            embedding =  detector.get_embeddings(face_img)
            face_tensor = preprocess_for_liveness(face_img).to(device)

            with torch.no_grad():
                output = liveness_detector(face_tensor)
                pred = torch.argmax(output, dim=1).item()
                is_real = (pred == 1)

            if not is_real:
                name = "FAKE"
                color = (0, 0, 255)
            else:
                name, dist, conf = recognizer.recognize(embedding)
                color = (0, 255, 0) if name != "Unknown" else (0, 255, 255)

            if name == "Unknown":
                label = f"Unknown{len(unknown_faces) + 1}"
                unknown_faces.append((embedding, face_img))
            else:
                label = name

            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            frame = draw_text_unicode(frame, label, (x1, y1 - 25), font_size=20, color=color)

        cv2.imshow("Face Recognition", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("a") and unknown_faces:
            print("\nNhập tên cho các khuôn mặt chưa biết:")
            for idx, (embedding, face_img) in enumerate(unknown_faces):
                window_name = f"Unknown Face {idx + 1}"
                bgr_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
                cv2.imshow(window_name, bgr_img)
                name = input(f"Nhập tên cho {window_name}: ").strip()
                if name:
                    database.add_face(name, embedding, filename=f"capture_{idx}.jpg")
                    print(f"Đã thêm {name} vào PostgreSQL.")
                cv2.destroyWindow(window_name)
            unknown_faces.clear()

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    database.close()


if __name__ == '__main__':
    main()
