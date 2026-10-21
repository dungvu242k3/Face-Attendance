"""Persistence operations for attendance logs."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from modules.database import Database


class AttendanceRepository:
    def __init__(self, database: Database):
        self.database = database

    def insert(
        self,
        connection: Any,
        employee_id: int,
        person_name: str,
        check_time: datetime,
        status: str,
        check_type: str,
        image_path: str | None = None,
    ) -> int:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO attendance_logs_512
                    (employee_id, person_name, check_time, status, image_path, check_type)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (employee_id, person_name, check_time, status, image_path, check_type),
            )
            return int(cursor.fetchone()[0])

    def find_recent(
        self,
        connection: Any,
        employee_id: int,
        check_type: str,
        since: datetime,
    ) -> tuple[Any, ...] | None:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, check_time, status
                FROM attendance_logs_512
                WHERE employee_id = %s
                  AND check_type = %s
                  AND check_time >= %s
                ORDER BY check_time DESC
                LIMIT 1
                """,
                (employee_id, check_type, since),
            )
            return cursor.fetchone()

    def lock_for_attendance(
        self, connection: Any, employee_id: int, check_type: str
    ) -> None:
        """Serialize concurrent requests for one employee/action pair."""

        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_advisory_xact_lock(%s, hashtext(%s))",
                (employee_id, check_type),
            )

    def count_between(self, start: datetime, end: datetime) -> int:
        with self.database.transaction() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM attendance_logs_512
                    WHERE check_time >= %s AND check_time < %s
                    """,
                    (start, end),
                )
                return int(cursor.fetchone()[0])

    def count_late_between(self, start: datetime, end: datetime) -> int:
        with self.database.transaction() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM attendance_logs_512
                    WHERE check_time >= %s AND check_time < %s
                      AND check_type = 'checkin'
                      AND status = 'late'
                    """,
                    (start, end),
                )
                return int(cursor.fetchone()[0])

    def recent(self, limit: int = 5) -> list[tuple[Any, ...]]:
        limit = max(1, min(int(limit), 200))
        with self.database.transaction() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT person_name, check_time, status, check_type
                    FROM attendance_logs_512
                    ORDER BY check_time DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                return cursor.fetchall()

    def search(
        self,
        name: str = "",
        check_type: str | None = None,
        limit: int = 200,
    ) -> list[tuple[Any, ...]]:
        limit = max(1, min(int(limit), 1000))
        query = """
            SELECT id, employee_id, person_name, check_time,
                   status, image_path, check_type
            FROM attendance_logs_512
            WHERE person_name ILIKE %s
        """
        params: list[Any] = [f"%{name}%"]
        if check_type in {"checkin", "checkout"}:
            query += " AND check_type = %s"
            params.append(check_type)
        query += " ORDER BY check_time DESC LIMIT %s"
        params.append(limit)

        with self.database.transaction() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                return cursor.fetchall()
