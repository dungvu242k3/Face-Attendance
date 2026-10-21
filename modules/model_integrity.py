"""Integrity checks for local machine-learning model artifacts."""

from __future__ import annotations

import hashlib
import hmac
from pathlib import Path


def verify_sha256(
    path: str | Path, expected: str | None, *, required: bool = False
) -> None:
    """Verify a model file before handing it to a deserializer.

    A hash does not make an untrusted model safe by itself, but it prevents a
    changed artifact from silently becoming executable application input. In
    production, model hashes are mandatory and supplied out-of-band.
    """

    if not expected:
        if required:
            raise ValueError(f"Thiếu SHA-256 cho model: {path}")
        return

    normalized = expected.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("SHA-256 model không hợp lệ")

    digest = hashlib.sha256()
    with Path(path).open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)

    if not hmac.compare_digest(digest.hexdigest(), normalized):
        raise ValueError(f"SHA-256 model không khớp: {path}")
