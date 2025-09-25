class InferenceEngine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, text):
        """Простейший инференс — возвращаем эмбеддинг текста"""
        tokens = self.tokenizer.encode(text)
        vector = self.model.predict(tokens)
        return f"[Ответ (вектор длины {len(vector)})]"
