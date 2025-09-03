class SimpleTrainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def train(self, texts):
        """Заглушка обучения — можно потом заменить на нормальную логику"""
        print("[Trainer] Начало обучения на", len(texts), "примеров")
        self.tokenizer.build_vocab(texts)
        print("[Trainer] Словарь построен:", len(self.tokenizer.vocab), "слов")
