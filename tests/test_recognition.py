import unittest

import numpy as np

from modules.recognition import FaceRecognizer


class FakeFaceRepository:
    def __init__(self, rows):
        self.rows = rows

    def search(self, embedding, top_k=1):
        self.received = embedding
        self.top_k = top_k
        return self.rows


class RecognitionTest(unittest.TestCase):
    def test_known_match_keeps_employee_id(self):
        repository = FakeFaceRepository([(7, "Alice", None, 0.2)])
        recognizer = FaceRecognizer(repository, threshold=0.6)
        result = recognizer.recognize(np.ones(512, dtype=np.float32))

        self.assertTrue(result.is_known)
        self.assertEqual(result.employee_id, 7)
        self.assertEqual(result.name, "Alice")
        self.assertEqual(repository.top_k, 1)

    def test_match_over_threshold_is_unknown(self):
        repository = FakeFaceRepository([(7, "Alice", None, 0.8)])
        result = FaceRecognizer(repository, threshold=0.6).recognize(
            np.ones(512, dtype=np.float32)
        )

        self.assertFalse(result.is_known)
        self.assertEqual(result.name, "Unknown")
        self.assertIsNone(result.employee_id)

    def test_invalid_embedding_is_rejected(self):
        recognizer = FaceRecognizer(FakeFaceRepository([]))
        with self.assertRaises(ValueError):
            recognizer.recognize(np.ones(128, dtype=np.float32))
