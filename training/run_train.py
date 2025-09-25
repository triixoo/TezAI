# training/run_train.py
import os
from datasets import load_from_disk, load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch

MODEL_NAME = os.getenv("MODEL_NAME", "distilgpt2")  
DATASET_PATH = "data/cleaned/"

def load_local_dataset():
    if os.path.exists(DATASET_PATH) and os.listdir(DATASET_PATH):
        print("Loading dataset from disk:", DATASET_PATH)
        ds = load_from_disk(DATASET_PATH)
        if isinstance(ds, Dataset):
            ds = ds.train_test_split(test_size=0.1)
        return ds

    
    csv_path = "data/raw/data.csv"
    if os.path.exists(csv_path):
        print("Loading CSV dataset:", csv_path)
        ds = load_dataset("csv", data_files=csv_path)["train"]
        if "text" not in ds.column_names:
            raise ValueError("CSV must have column 'text'")
        ds = ds.train_test_split(test_size=0.1)
        os.makedirs(DATASET_PATH, exist_ok=True)
        ds.save_to_disk(DATASET_PATH)
        print("Saved cleaned dataset to", DATASET_PATH)
        return ds

    # 3) fallback dummy dataset
    print("No dataset found — creating dummy dataset.")
    texts = [
        "Привет, меня зовут TezAI.",
        "Сегодня отличная погода и мы учим модель.",
        "Как дела? Я могу ответить на вопросы.",
        "Тренировка завершится быстро на dummy данных."
    ]
    ds = Dataset.from_dict({"text": texts})
    ds = ds.train_test_split(test_size=0.25)
    return ds

def tokenize_batch(batch, tokenizer):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

def add_labels(batch):
    # labels = input_ids for causal LM
    batch["labels"] = batch["input_ids"].copy()
    return batch

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ds = load_local_dataset()

    print("Model:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # If tokenizer has no pad token, set it
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))

    # tokenize
    ds = ds.map(lambda x: tokenize_batch(x, tokenizer), batched=True, remove_columns=[c for c in ds["train"].column_names if c!="text"])
    # add labels
    ds = ds.map(add_labels, batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_dataset = ds["train"]
    eval_dataset = ds["test"] if "test" in ds else None

    args = TrainingArguments(
        output_dir="./checkpoints",
        per_device_train_batch_size=1,          # small because VRAM limited
        gradient_accumulation_steps=4,          # simulate larger batch
        num_train_epochs=1,
        fp16=torch.cuda.is_available(),        # use mixed precision if CUDA
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    os.makedirs("./models/tezai", exist_ok=True)
    model.save_pretrained("./models/tezai")
    tokenizer.save_pretrained("./models/tezai")
    print("Training finished. Model saved to ./models/tezai")

if __name__ == "__main__":
    train()
