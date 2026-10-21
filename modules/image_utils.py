"""Safe image decoding helpers for browser uploads and data URLs."""

from __future__ import annotations

import base64
import binascii
import struct

import cv2
import numpy as np


class InvalidImage(ValueError):
    """Raised when an uploaded image cannot be decoded safely."""


def decode_data_url(
    value: object,
    max_bytes: int,
    max_pixels: int = 4_000_000,
    max_dimension: int = 4096,
) -> np.ndarray:
    if not isinstance(value, str) or not value.strip():
        raise InvalidImage("Ảnh không được để trống")

    encoded = (
        value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
    )
    encoded = encoded.strip()
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidImage("Dữ liệu ảnh không hợp lệ") from exc

    return decode_image_bytes(
        image_bytes,
        max_bytes=max_bytes,
        max_pixels=max_pixels,
        max_dimension=max_dimension,
    )


def decode_image_bytes(
    image_bytes: bytes,
    max_bytes: int,
    max_pixels: int = 4_000_000,
    max_dimension: int = 4096,
) -> np.ndarray:
    if not image_bytes:
        raise InvalidImage("Ảnh không được để trống")
    if len(image_bytes) > max_bytes:
        raise InvalidImage(f"Ảnh vượt quá {max_bytes // (1024 * 1024)}MB")

    dimensions = _read_image_dimensions(image_bytes)
    if dimensions is None:
        raise InvalidImage("Định dạng ảnh không được hỗ trợ")
    width, height = dimensions
    if (
        width <= 0
        or height <= 0
        or width > max_dimension
        or height > max_dimension
        or width * height > max_pixels
    ):
        raise InvalidImage("Kích thước ảnh vượt quá giới hạn cho phép")

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise InvalidImage("Ảnh không hợp lệ")
    return image


def _read_image_dimensions(image_bytes: bytes) -> tuple[int, int] | None:
    """Read dimensions before native decoding to prevent pixel-count bombs."""

    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        if len(image_bytes) < 24 or image_bytes[12:16] != b"IHDR":
            return None
        return struct.unpack(">II", image_bytes[16:24])

    if image_bytes.startswith(b"\xff\xd8"):
        return _read_jpeg_dimensions(image_bytes)

    if (
        len(image_bytes) >= 16
        and image_bytes[:4] == b"RIFF"
        and image_bytes[8:12] == b"WEBP"
    ):
        return _read_webp_dimensions(image_bytes)

    return None


def _read_jpeg_dimensions(image_bytes: bytes) -> tuple[int, int] | None:
    index = 2
    sof_markers = (
        set(range(0xC0, 0xC4))
        | set(range(0xC5, 0xC8))
        | set(range(0xC9, 0xCC))
        | set(range(0xCD, 0xD0))
    )

    while index + 1 < len(image_bytes):
        if image_bytes[index] != 0xFF:
            index += 1
            continue
        while index < len(image_bytes) and image_bytes[index] == 0xFF:
            index += 1
        if index >= len(image_bytes):
            return None

        marker = image_bytes[index]
        index += 1
        if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            continue
        if index + 2 > len(image_bytes):
            return None

        segment_length = struct.unpack(">H", image_bytes[index : index + 2])[0]
        if segment_length < 2 or index + segment_length > len(image_bytes):
            return None
        if marker in sof_markers and segment_length >= 7:
            height, width = struct.unpack(">HH", image_bytes[index + 3 : index + 7])
            return width, height
        index += segment_length

    return None


def _read_webp_dimensions(image_bytes: bytes) -> tuple[int, int] | None:
    chunk = image_bytes[12:16]
    if chunk == b"VP8X" and len(image_bytes) >= 30:
        width = 1 + int.from_bytes(image_bytes[24:27], "little")
        height = 1 + int.from_bytes(image_bytes[27:30], "little")
        return width, height
    if (
        chunk == b"VP8 "
        and len(image_bytes) >= 30
        and image_bytes[23:26] == b"\x9d\x01\x2a"
    ):
        width = struct.unpack("<H", image_bytes[26:28])[0] & 0x3FFF
        height = struct.unpack("<H", image_bytes[28:30])[0] & 0x3FFF
        return width, height
    if chunk == b"VP8L" and len(image_bytes) >= 26 and image_bytes[20] == 0x2F:
        width = 1 + ((image_bytes[21] | image_bytes[22] << 8) & 0x3FFF)
        height = 1 + (
            (image_bytes[22] >> 6 | image_bytes[23] << 2 | image_bytes[24] << 10)
            & 0x3FFF
        )
        return width, height
    return None
