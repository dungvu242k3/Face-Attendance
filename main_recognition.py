"""Optional local webcam recognition client."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import Settings
from modules.camera import Camera
from modules.service_container import build_services


def draw_text_unicode(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    font_size: int = 20,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(image)
    rgb_color = (color[2], color[1], color[0])
    draw.text(position, text, font=font, fill=rgb_color)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def _register_unknown(
    services, unknown_faces: list[tuple[np.ndarray, np.ndarray]]
) -> None:
    print("\nNhập tên cho các khuôn mặt chưa biết:")
    for index, (embedding, face_img) in enumerate(unknown_faces, start=1):
        window_name = f"Unknown Face {index}"
        cv2.imshow(window_name, face_img)
        name = input(f"Nhập tên cho {window_name}: ").strip()
        if name:
            services.employee_service.create_with_face(
                name=name,
                email=None,
                phone=None,
                position=None,
                embedding=embedding,
                filename=f"capture_{index}.jpg",
            )
            print(f"Đã thêm {name} vào PostgreSQL.")
        cv2.destroyWindow(window_name)


def main() -> None:
    settings = Settings.from_env()
    services = build_services(settings, with_liveness=True)
    camera = Camera()
    unknown_faces: list[tuple[np.ndarray, np.ndarray]] = []

    try:
        while True:
            ok, frame = camera.read_frame()
            if not ok:
                break

            boxes = services.detector.detect_faces(frame)
            for box in boxes:
                face_img = services.detector.crop_face(frame, box)
                if face_img is None:
                    continue

                if not services.liveness.is_real(face_img):
                    name = "FAKE"
                    color = (0, 0, 255)
                else:
                    embedding = services.detector.get_embeddings(face_img)
                    match = services.recognizer.recognize(embedding)
                    if match.is_known:
                        name = match.name
                        color = (0, 255, 0)
                    else:
                        name = f"Unknown{len(unknown_faces) + 1}"
                        color = (0, 255, 255)
                        unknown_faces.append((embedding, face_img.copy()))

                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                frame = draw_text_unicode(
                    frame, name, (x1, max(0, y1 - 25)), color=color
                )

            cv2.imshow("Face Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("a") and unknown_faces:
                _register_unknown(services, unknown_faces)
                unknown_faces.clear()
            elif key == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        services.database.close()


if __name__ == "__main__":
    main()
