"""OpenCV camera adapter used by the local webcam client."""

from __future__ import annotations

import cv2


class Camera:
    def __init__(self, src: int = 0, width: int = 1280, height: int = 720):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Không thể mở camera {src}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read_frame(self):
        return self.cap.read()

    def release(self) -> None:
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
