"""Face detection and FaceNet embedding generation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1

from modules.model_integrity import verify_sha256


class FaceDetector:
    """Adapter around the YOLO face detector and FaceNet embedder."""

    def __init__(
        self,
        model_path: str | Path,
        backend: str = "haar",
        device: str = "auto",
        confidence: float = 0.5,
        face_class_ids: Iterable[int] | None = None,
        model_sha256: str | None = None,
        require_model_sha256: bool = False,
    ):
        self.backend = (backend or "haar").lower()
        if self.backend not in {"haar", "yolo"}:
            raise ValueError("DETECTOR_BACKEND phải là haar hoặc yolo")
        self.device = self._resolve_device(device)
        self.confidence = confidence
        self.face_class_ids = (
            list(face_class_ids) if face_class_ids is not None else None
        )

        self.detector = None
        self.face_cascade = None
        if self.backend == "yolo":
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Không tìm thấy YOLO model: {model_path}")
            verify_sha256(
                model_path,
                model_sha256,
                required=require_model_sha256,
            )
            from ultralytics import YOLO

            self.detector = YOLO(str(model_path))
            self.detector.to(self.device)
            self._validate_face_model()
        else:
            cascade_path = (
                Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            )
            self.face_cascade = cv2.CascadeClassifier(str(cascade_path))
            if self.face_cascade.empty():
                raise RuntimeError("Không thể tải Haar face detector")
        self.embedder = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    def _validate_face_model(self) -> None:
        """Prevent a generic multi-class detector from being used as face-only."""

        if self.face_class_ids is not None:
            return
        names = getattr(self.detector, "names", {})
        values = list(names.values()) if isinstance(names, dict) else list(names or [])
        if len(values) > 1:
            labels = {str(value).lower() for value in values}
            if not any("face" in label for label in labels):
                raise ValueError(
                    "YOLO model không phải detector khuôn mặt rõ ràng; "
                    "hãy đặt YOLO_MODEL_PATH tới model face hoặc cấu hình YOLO_FACE_CLASS_IDS"
                )

    @staticmethod
    def _resolve_device(device: str) -> str:
        normalized = (device or "auto").lower()
        if normalized == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if normalized == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("DEVICE=cuda nhưng máy không có CUDA khả dụng")
        if normalized not in {"cpu", "cuda"}:
            raise ValueError("DEVICE phải là auto, cpu hoặc cuda")
        return normalized

    def detect_faces(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        if frame is None or frame.size == 0:
            return []

        if self.backend == "haar":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40),
            )
            return [
                (int(x), int(y), int(x + width), int(y + height))
                for x, y, width, height in detected
            ]

        kwargs = {
            "source": frame,
            "conf": self.confidence,
            "device": self.device,
            "verbose": False,
        }
        if self.face_class_ids is not None:
            kwargs["classes"] = self.face_class_ids

        results = self.detector.predict(**kwargs)[0]
        boxes: list[tuple[int, int, int, int]] = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            boxes.append((x1, y1, x2, y2))
        return boxes

    def crop_face(
        self,
        frame: np.ndarray,
        box: tuple[int, int, int, int],
        size: int = 160,
    ) -> np.ndarray | None:
        if frame is None or frame.size == 0:
            return None

        height, width = frame.shape[:2]
        x1, y1, x2, y2 = box
        x1 = max(0, min(int(x1), width - 1))
        y1 = max(0, min(int(y1), height - 1))
        x2 = max(x1 + 1, min(int(x2), width))
        y2 = max(y1 + 1, min(int(y2), height))
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return None
        return cv2.resize(face, (size, size), interpolation=cv2.INTER_AREA)

    def extract_faces(
        self,
        frame: np.ndarray,
        boxes: Iterable[tuple[int, int, int, int]],
        size: int = 160,
    ) -> list[np.ndarray]:
        return [
            face
            for box in boxes
            if (face := self.crop_face(frame, box, size=size)) is not None
        ]

    def preprocess(self, face_img: np.ndarray) -> torch.Tensor:
        face = cv2.resize(face_img, (160, 160), interpolation=cv2.INTER_AREA)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype(np.float32) / 255.0
        face = (face - 0.5) / 0.5
        return torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def get_embeddings(self, face_img: np.ndarray) -> np.ndarray:
        face_tensor = self.preprocess(face_img)
        with torch.inference_mode():
            embedding = self.embedder(face_tensor).cpu().numpy()[0]

        norm = np.linalg.norm(embedding)
        if norm == 0:
            raise ValueError("FaceNet trả về embedding rỗng")
        return embedding / norm
