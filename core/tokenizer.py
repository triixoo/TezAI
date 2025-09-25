class SimpleTokenizer:
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}

    def build_vocab(self, texts):
        """Создание словаря на основе датасета"""
        tokens = set(" ".join(texts).split())
        self.vocab = {tok: i for i, tok in enumerate(tokens, start=1)}
        self.vocab["<PAD>"] = 0
        self.inverse_vocab = {i: tok for tok, i in self.vocab.items()}

    def encode(self, text):
        """Преобразуем текст в список индексов"""
        return [self.vocab.get(tok, 0) for tok in text.split()]

    def decode(self, tokens):
        """Преобразуем индексы обратно в текст"""
        return " ".join([self.inverse_vocab.get(i, "<UNK>") for i in tokens])