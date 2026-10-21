"""Composition root for application services."""

from __future__ import annotations

from dataclasses import dataclass

from config import Settings
from modules.attendance_service import AttendanceService
from modules.database import Database
from modules.detector import FaceDetector
from modules.employee_service import EmployeeService
from modules.liveness import LivenessDetector
from modules.recognition import FaceRecognizer
from modules.recognition_service import FaceRecognitionService
from modules.repositories.attendance_repository import AttendanceRepository
from modules.repositories.employee_repository import EmployeeRepository
from modules.repositories.face_repository import FaceEmbeddingRepository


@dataclass
class Services:
    settings: Settings
    database: Database
    employees: EmployeeRepository
    faces: FaceEmbeddingRepository
    attendance: AttendanceRepository
    employee_service: EmployeeService
    attendance_service: AttendanceService
    detector: FaceDetector
    liveness: LivenessDetector | None = None
    recognizer: FaceRecognizer | None = None
    recognition_service: FaceRecognitionService | None = None


def build_services(settings: Settings, with_liveness: bool) -> Services:
    database = Database(settings)
    employees = EmployeeRepository(database)
    faces = FaceEmbeddingRepository(database)
    attendance = AttendanceRepository(database)
    employee_service = EmployeeService(database, employees, faces)
    attendance_service = AttendanceService(database, attendance, settings)
    detector = FaceDetector(
        model_path=settings.yolo_model_path,
        backend=settings.detector_backend,
        device=settings.device,
        confidence=settings.yolo_confidence,
        face_class_ids=settings.yolo_face_class_ids,
        model_sha256=settings.yolo_model_sha256,
        require_model_sha256=settings.is_production,
    )

    if not with_liveness:
        return Services(
            settings=settings,
            database=database,
            employees=employees,
            faces=faces,
            attendance=attendance,
            employee_service=employee_service,
            attendance_service=attendance_service,
            detector=detector,
        )

    liveness = LivenessDetector(
        model_path=settings.liveness_model_path,
        device=settings.device,
        real_class=settings.liveness_real_class,
        real_threshold=settings.liveness_threshold,
        model_sha256=settings.liveness_model_sha256,
        require_model_sha256=settings.is_production,
    )
    recognizer = FaceRecognizer(faces, threshold=settings.recognition_threshold)
    recognition_service = FaceRecognitionService(detector, liveness, recognizer)
    return Services(
        settings=settings,
        database=database,
        employees=employees,
        faces=faces,
        attendance=attendance,
        employee_service=employee_service,
        attendance_service=attendance_service,
        detector=detector,
        liveness=liveness,
        recognizer=recognizer,
        recognition_service=recognition_service,
    )
