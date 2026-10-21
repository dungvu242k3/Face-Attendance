"""Validation and serialization helpers for face embeddings."""

from __future__ import annotations

from typing import Any

import numpy as np


def as_embedding_array(embedding: Any) -> np.ndarray:
    """Return a finite, one-dimensional 512-value float32 embedding."""

    if hasattr(embedding, "detach"):
        embedding = embedding.detach().cpu().numpy()

    array = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if array.shape != (512,):
        raise ValueError(f"Embedding must contain 512 values, received {array.size}")
    if not np.isfinite(array).all():
        raise ValueError("Embedding contains NaN or infinite values")
    return array


def embedding_to_vector(embedding: Any) -> str:
    """Serialize an embedding in the format accepted by pgvector."""

    array = as_embedding_array(embedding)
    return "[" + ",".join(format(float(value), ".9g") for value in array) + "]"
