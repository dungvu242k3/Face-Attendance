"""Face embedding matching use case."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from modules.embedding_utils import as_embedding_array


class FaceSearchRepository(Protocol):
    def search(self, embedding: Any, top_k: int = 1) -> list[tuple[Any, ...]]: ...


@dataclass(frozen=True)
class RecognitionMatch:
    employee_id: int | None
    name: str
    distance: float | None
    confidence: float | None
    filename: str | None = None

    @property
    def is_known(self) -> bool:
        return self.name != "Unknown" and self.employee_id is not None


class FaceRecognizer:
    def __init__(self, database: FaceSearchRepository, threshold: float = 0.6):
        self.database = database
        self.threshold = threshold

    def recognize(self, embedding: Any) -> RecognitionMatch:
        if embedding is None:
            return RecognitionMatch(None, "Unknown", None, None)

        embedding_array = as_embedding_array(embedding)
        results = self.database.search(embedding_array, top_k=1)
        if not results:
            return RecognitionMatch(None, "Unknown", None, None)

        employee_id, person_name, filename, raw_distance = results[0]
        distance = float(raw_distance)
        confidence = max(0.0, 1.0 - distance)
        if distance <= self.threshold:
            return RecognitionMatch(
                employee_id=employee_id,
                name=person_name,
                distance=distance,
                confidence=confidence,
                filename=filename,
            )
        return RecognitionMatch(
            employee_id=None,
            name="Unknown",
            distance=distance,
            confidence=confidence,
            filename=filename,
        )
