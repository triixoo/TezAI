from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_from_disk

MODEL_NAME = "gpt2"
DATASET_PATH = "data/cleaned/"

def train():
    dataset = load_from_disk(DATASET_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    dataset = dataset.map(tokenize, batched=True)

    args = TrainingArguments(
        output_dir="./checkpoints",
        evaluation_strategy="steps",
        per_device_train_batch_size=2,
        num_train_epochs=1,
        save_steps=500,
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )

    trainer.train()
    model.save_pretrained("./models/tezai")
    tokenizer.save_pretrained("./models/tezai")

if __name__ == "__main__":
    train()