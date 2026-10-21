"""Persistence operations for employees."""

from __future__ import annotations

from typing import Any

from modules.database import Database


class EmployeeRepository:
    def __init__(self, database: Database):
        self.database = database

    def count(self) -> int:
        with self.database.transaction() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM employees")
                return int(cursor.fetchone()[0])

    def list(self, name: str = "", position: str = "") -> list[tuple[Any, ...]]:
        query = """
            SELECT id, name, email, phone, position, created_at
            FROM employees
            WHERE name ILIKE %s
        """
        params: list[Any] = [f"%{name}%"]
        if position:
            query += " AND position = %s"
            params.append(position)
        query += " ORDER BY created_at DESC, id DESC"

        with self.database.transaction() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                return cursor.fetchall()

    def positions(self) -> list[str]:
        with self.database.transaction() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT DISTINCT position
                    FROM employees
                    WHERE position IS NOT NULL AND BTRIM(position) <> ''
                    ORDER BY position
                    """
                )
                return [row[0] for row in cursor.fetchall()]

    def get(self, employee_id: int) -> tuple[Any, ...] | None:
        with self.database.transaction() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, name, email, phone, position, created_at
                    FROM employees
                    WHERE id = %s
                    """,
                    (employee_id,),
                )
                return cursor.fetchone()

    def insert(
        self,
        connection: Any,
        name: str,
        email: str | None,
        phone: str | None,
        position: str | None,
    ) -> int:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO employees (name, email, phone, position)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (name, email, phone, position),
            )
            return int(cursor.fetchone()[0])

    def update(
        self,
        connection: Any,
        employee_id: int,
        name: str,
        email: str | None,
        phone: str | None,
        position: str | None,
    ) -> bool:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE employees
                SET name = %s, email = %s, phone = %s, position = %s
                WHERE id = %s
                """,
                (name, email, phone, position, employee_id),
            )
            return cursor.rowcount == 1

    def delete(self, connection: Any, employee_id: int) -> bool:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM employees WHERE id = %s", (employee_id,))
            return cursor.rowcount == 1
