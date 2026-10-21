"""Use cases for employee and face-profile management."""

from __future__ import annotations

from typing import Any

from modules.database import Database
from modules.repositories.employee_repository import EmployeeRepository
from modules.repositories.face_repository import FaceEmbeddingRepository


class EmployeeService:
    def __init__(
        self,
        database: Database,
        employees: EmployeeRepository,
        faces: FaceEmbeddingRepository,
    ):
        self.database = database
        self.employees = employees
        self.faces = faces

    def create_with_face(
        self,
        name: str,
        email: str | None,
        phone: str | None,
        position: str | None,
        embedding: Any,
        filename: str | None = None,
    ) -> int:
        name = self._normalize_name(name)
        with self.database.transaction() as connection:
            employee_id = self.employees.insert(
                connection, name, email, phone, position
            )
            self.faces.insert(connection, employee_id, name, embedding, filename)
            return employee_id

    def update(
        self,
        employee_id: int,
        name: str,
        email: str | None,
        phone: str | None,
        position: str | None,
    ) -> bool:
        name = self._normalize_name(name)
        with self.database.transaction() as connection:
            updated = self.employees.update(
                connection, employee_id, name, email, phone, position
            )
            if not updated:
                return False
            self.faces.update_person_name(connection, employee_id, name)
            return True

    def delete(self, employee_id: int) -> bool:
        with self.database.transaction() as connection:
            return self.employees.delete(connection, employee_id)

    @staticmethod
    def _normalize_name(name: str | None) -> str:
        normalized = (name or "").strip()
        if not normalized:
            raise ValueError("Tên nhân viên là bắt buộc")
        return normalized
