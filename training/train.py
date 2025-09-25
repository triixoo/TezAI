import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import tensorflow_datasets as tfds
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.tezai.tezai_lm import TezAILM

MODEL_NAME = "gpt2"
BATCH_SIZE = 8
EPOCHS = 1
MAX_LEN = 128
LR = 5e-4


datasets_to_load = [
    ("wiki_qa", "train"),
    ("trec", "train"),
    ("yahoo_answers_topics", "train"),


    ("ubuntu_dialogs_corpus", "train"),
    ("customer_support_tweets", "train"),

    
    ("daily_dialog", "train"),
    ("conv_ai_2", "train"),
    ("cornell_movie_dialog", "train"),
    ("multi_woz_v22", "train"),
    ("taskmaster1", "train"),
    ("taskmaster2", "train"),
]

def load_working_dataset(ds_name, split="train"):
    try:
        ds = load_dataset(ds_name, split=split)
        print(f"✅ Загружен через 🤗 datasets: {ds_name} split={split}")
        return ds
    except Exception as e1:
        print(f"⚠️ Не удалось через 🤗 datasets: {ds_name}, {e1}")
        try:
            tfds_name = f"huggingface:{ds_name}"
            ds_tf = tfds.load(tfds_name, split=split)
            print(f"✅ Загружен через TFDS: {tfds_name}")
            from datasets import Dataset
            data = []
            for ex in tfds.as_numpy(ds_tf):
                data.append(ex)
            return Dataset.from_list(data)
        except Exception as e2:
            print(f"❌ Не удалось через TFDS: {ds_name}, {e2}")
            return None

def extract_dialog(item):
    """универсальный парсер для разных датасетов"""
    if "dialog" in item:
        return item["dialog"]
    if "utterances" in item and isinstance(item["utterances"], list):
        return [u["text"] if isinstance(u, dict) else str(u) for u in item["utterances"]]
    if "conversation" in item:
        return item["conversation"]
    if "text" in item and "response" in item:
        return [item["text"], item["response"]]
    if "question" in item and "answer" in item:
        return [item["question"], item["answer"]]
    if "utterance" in item:
        return [item["utterance"]]
    return None

def train():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs, labels = [], []

    # ====== загружаем все датасеты ======
    for ds_name, split in datasets_to_load:
        ds = load_working_dataset(ds_name, split)
        if ds is None:
            continue

        for item in ds:
            dialog = extract_dialog(item)
            if dialog and isinstance(dialog, list):
                for i in range(len(dialog) - 1):
                    q = str(dialog[i])
                    a = str(dialog[i + 1])
                    text = q + tokenizer.eos_token + a + tokenizer.eos_token
                    enc = tokenizer(
                        text,
                        truncation=True,
                        padding="max_length",
                        max_length=MAX_LEN,
                        return_tensors="pt"
                    )
                    inputs.append(enc["input_ids"][0])
                    labels.append(enc["input_ids"][0])

    if len(inputs) == 0:
        print("❌ Нет данных для обучения! Проверьте доступность датасетов или добавьте свой dialogue.csv.")
        return

    class DialogueDataset(torch.utils.data.Dataset):
        def __init__(self, inputs, labels):
            self.inputs = inputs
            self.labels = labels
        def __len__(self):
            return len(self.inputs)
        def __getitem__(self, idx):
            return {"input_ids": self.inputs[idx], "labels": self.labels[idx]}

    dataset = DialogueDataset(inputs, labels)
    vocab_size = tokenizer.vocab_size

    model = TezAILM(vocab_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            labels_batch = batch["labels"].to(device)

            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, vocab_size), labels_batch.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step % 10 == 0:
                print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

    save_path = "./models/tezai_lm"
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
    tokenizer.save_pretrained(save_path)
    required_files = [
        "tokenizer.json",
        "vocab.json",
        "tokenizer_config.json",
        "special_tokens_map.json"
    ]
    missing = [f for f in required_files if not os.path.isfile(os.path.join(save_path, f))]
    if missing:
        print(f"❌ Не все файлы токенайзера сохранены: {missing}")
    else:
        print(f"✅ Model and tokenizer saved to {save_path}")

if __name__ == "__main__":
    train()
