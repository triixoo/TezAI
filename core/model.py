import numpy as np

class SimpleModel:
    def __init__(self, vocab_size, embedding_dim=16):
        self.embeddings = np.random.rand(vocab_size, embedding_dim)

    def forward(self, input_ids):
        """Берём эмбеддинги слов"""
        return np.mean([self.embeddings[idx] for idx in input_ids if idx < len(self.embeddings)], axis=0)

    def predict(self, input_ids):
        """Простейший предикт — возвращаем среднее эмбеддингов"""
        return self.forward(input_ids)
