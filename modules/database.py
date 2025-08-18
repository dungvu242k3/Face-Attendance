import numpy as np
import psycopg2


class FaceDatabase:
    def __init__(self, table_name="face_embeddings_512",
                 dbname="face_db", user="postgres", password="sat24042003", host="localhost", port=5432):
        self.table_name = table_name
        self.conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )
        self.cur = self.conn.cursor()

    def add_face(self, person_name, embedding, filename=None):
        if isinstance(embedding, np.ndarray):
            embedding_str = "[" + ",".join(map(str, embedding.astype(np.float32))) + "]"
        else:
            embedding_str = "[" + ",".join(map(str, embedding.cpu().numpy().astype(np.float32))) + "]"

        self.cur.execute(
            f"INSERT INTO {self.table_name} (person_name, embedding, filename) VALUES (%s, %s, %s)",
            (person_name, embedding_str, filename)
        )
        self.conn.commit()

    def search_face(self, embedding, top_k=1):
        if isinstance(embedding, np.ndarray):
            embedding_str = "[" + ",".join(map(str, embedding.astype(np.float32))) + "]"
        else:
            embedding_str = "[" + ",".join(map(str, embedding.cpu().numpy().astype(np.float32))) + "]"

        self.cur.execute(
            f"""
            SELECT person_name, filename, embedding <=> %s AS distance
            FROM {self.table_name}
            ORDER BY distance ASC
            LIMIT {top_k}
            """,
            (embedding_str,)
        )
        return self.cur.fetchall()

    def close(self):
        self.cur.close()
        self.conn.close()
