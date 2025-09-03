from typing import List, Tuple, Any
import numpy as np
import faiss


class VectorDB:
    def __init__(self, dim: int):
        """
        dim: размерность эмбеддингов
        """
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)  # простой L2 индекс
        self.vectors = []  # для хранения векторов
        self.metadata = []  # для хранения связанной инфы (id, текст и т.д.)

    def add(self, vector: np.ndarray, meta: Any):
        """
        Добавить новый вектор в базу.
        vector: np.ndarray размерности (dim,)
        meta: любая метаинформация (например текст документа или id)
        """
        if vector.shape != (self.dim,):
            raise ValueError(f"Vector shape must be ({self.dim},), got {vector.shape}")

        self.index.add(vector.reshape(1, -1))
        self.vectors.append(vector)
        self.metadata.append(meta)

    def search(self, query: np.ndarray, k: int = 5) -> List[Tuple[float, Any]]:
        """
        Найти k ближайших векторов
        query: вектор запроса (np.ndarray размерности (dim,))
        return: список из (distance, meta)
        """
        if query.shape != (self.dim,):
            raise ValueError(f"Query shape must be ({self.dim},), got {query.shape}")

        distances, indices = self.index.search(query.reshape(1, -1), k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append((float(dist), self.metadata[idx]))
        return results
