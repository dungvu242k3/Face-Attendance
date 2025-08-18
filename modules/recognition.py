from turtle import distance

import numpy as np


class FaceRecognizer:
    def __init__(self, database, threshold=0.7):
        """
        database: đối tượng FaceDatabase
        threshold: ngưỡng nhận diện (cosine distance)
        """
        self.database = database
        self.threshold = threshold

    def recognize(self, embedding):
        if embedding is None:
            return None, "Unknown", None, None

        if not isinstance(embedding, np.ndarray):
            if hasattr(embedding, 'cpu'):
                embedding = embedding.cpu().numpy()
            else:
                embedding = np.array(embedding)

        results = self.database.search_face(embedding, top_k=1)
        if not results:
            return None, "Unknown", None, None

    # results[0] có thể chỉ trả về 3 giá trị
        if len(results[0]) == 3:
            person_name, filename, distance = results[0]
            employee_id = None
        elif len(results[0]) == 4:
            employee_id, person_name, filename, distance = results[0]
        else:
        # trường hợp lạ, trả về mặc định
            return None, "Unknown", None, None

        if distance <= self.threshold:
            confidence = 1 - distance
            return employee_id, person_name, distance, confidence
        else:
            return None, "Unknown", distance, 1 - distance


