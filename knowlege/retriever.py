import numpy as np
from typing import List, Tuple, Any
from .vector_db import VectorDB


class Retriever:
    def __init__(self, dim: int):
        self.db = VectorDB(dim)

    def add_document(self, embedding: np.ndarray, text: str):
        """
        Добавляем текст в базу
        """
        self.db.add(embedding, text)

    def retrieve(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[float, Any]]:
        """
        Ищем релевантные документы
        """
        return self.db.search(query_embedding, k)
