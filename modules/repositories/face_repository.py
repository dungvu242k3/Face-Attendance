"""Persistence operations for face embeddings."""

from __future__ import annotations

from typing import Any

from modules.database import Database
from modules.embedding_utils import embedding_to_vector


class FaceEmbeddingRepository:
    def __init__(self, database: Database):
        self.database = database

    def insert(
        self,
        connection: Any,
        employee_id: int | None,
        person_name: str,
        embedding: Any,
        filename: str | None = None,
    ) -> int:
        vector = embedding_to_vector(embedding)
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO face_embeddings_512
                    (employee_id, person_name, embedding, filename)
                VALUES (%s, %s, %s::vector, %s)
                RETURNING id
                """,
                (employee_id, person_name, vector, filename),
            )
            return int(cursor.fetchone()[0])

    def search(self, embedding: Any, top_k: int = 1) -> list[tuple[Any, ...]]:
        if not isinstance(top_k, int) or not 1 <= top_k <= 100:
            raise ValueError("top_k must be an integer between 1 and 100")
        vector = embedding_to_vector(embedding)

        with self.database.transaction() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT employee_id, person_name, filename,
                           embedding <=> %s::vector AS distance
                    FROM face_embeddings_512
                    ORDER BY distance ASC
                    LIMIT %s
                    """,
                    (vector, top_k),
                )
                return cursor.fetchall()

    def update_person_name(self, connection: Any, employee_id: int, name: str) -> None:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE face_embeddings_512
                SET person_name = %s
                WHERE employee_id = %s
                """,
                (name, employee_id),
            )
