import hashlib
import tempfile
import unittest
from pathlib import Path

from modules.model_integrity import verify_sha256


class ModelIntegrityTest(unittest.TestCase):
    def test_matching_hash_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.bin"
            path.write_bytes(b"trusted model")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()

            verify_sha256(path, digest, required=True)

    def test_changed_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.bin"
            path.write_bytes(b"tampered model")

            with self.assertRaises(ValueError):
                verify_sha256(path, "0" * 64, required=True)

    def test_required_hash_cannot_be_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.bin"
            path.write_bytes(b"model")

            with self.assertRaises(ValueError):
                verify_sha256(path, None, required=True)
