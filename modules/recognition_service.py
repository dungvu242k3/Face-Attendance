"""Orchestrates detection, liveness and identity matching."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from modules.detector import FaceDetector
from modules.liveness import LivenessDetector
from modules.recognition import FaceRecognizer, RecognitionMatch


@dataclass(frozen=True)
class FaceRecognitionResult:
    status: str
    message: str
    name: str
    employee_id: int | None = None
    embedding: np.ndarray | None = None
    distance: float | None = None
    confidence: float | None = None


class FaceRecognitionService:
    def __init__(
        self,
        detector: FaceDetector,
        liveness: LivenessDetector,
        recognizer: FaceRecognizer,
    ):
        self.detector = detector
        self.liveness = liveness
        self.recognizer = recognizer

    def recognize_frame(self, frame: np.ndarray) -> FaceRecognitionResult:
        boxes = self.detector.detect_faces(frame)
        crops = [
            (box, face)
            for box in boxes
            if (face := self.detector.crop_face(frame, box)) is not None
        ]

        if not crops:
            return FaceRecognitionResult(
                status="no_face",
                message="Không phát hiện khuôn mặt",
                name="Unknown",
            )
        if len(crops) > 1:
            return FaceRecognitionResult(
                status="multiple_faces",
                message="Vui lòng chỉ để một khuôn mặt trong khung hình",
                name="Unknown",
            )

        _, face_img = crops[0]
        if not self.liveness.is_real(face_img):
            return FaceRecognitionResult(
                status="fake",
                message="Phát hiện ảnh giả",
                name="FAKE",
            )

        embedding = self.detector.get_embeddings(face_img)
        match = self.recognizer.recognize(embedding)
        if not match.is_known:
            return FaceRecognitionResult(
                status="unknown",
                message="Không nhận diện được",
                name="Unknown",
                embedding=embedding,
                distance=match.distance,
                confidence=match.confidence,
            )

        return self._known_result(match, embedding)

    @staticmethod
    def _known_result(
        match: RecognitionMatch,
        embedding: np.ndarray,
    ) -> FaceRecognitionResult:
        return FaceRecognitionResult(
            status="ok",
            message="Nhận diện thành công",
            name=match.name,
            employee_id=match.employee_id,
            embedding=embedding,
            distance=match.distance,
            confidence=match.confidence,
        )
