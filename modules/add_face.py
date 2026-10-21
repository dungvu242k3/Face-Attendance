"""Local webcam utility for creating an employee face profile."""

from __future__ import annotations

import cv2

from config import Settings
from modules.camera import Camera
from modules.service_container import build_services


def main() -> None:
    name = input("Nhập tên nhân viên: ").strip()
    if not name:
        print("Tên nhân viên là bắt buộc")
        return

    settings = Settings.from_env()
    services = build_services(settings, with_liveness=False)
    camera = Camera()

    try:
        while True:
            ok, frame = camera.read_frame()
            if not ok:
                break

            boxes = services.detector.detect_faces(frame)
            if len(boxes) == 1:
                face_img = services.detector.crop_face(frame, boxes[0])
                if face_img is not None:
                    embedding = services.detector.get_embeddings(face_img)
                    services.employee_service.create_with_face(
                        name=name,
                        email=None,
                        phone=None,
                        position=None,
                        embedding=embedding,
                        filename="webcam_capture.jpg",
                    )
                    print(f"Đã thêm: {name}")
                    break
            elif len(boxes) > 1:
                print("Vui lòng chỉ để một khuôn mặt trong khung hình")

            cv2.imshow("Add Face", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        services.database.close()


if __name__ == "__main__":
    main()
