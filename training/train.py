# training/train.py
from training.dataset import load_dataset, preprocess_data

def main():
    dataset = load_dataset("data/raw/")
    train_data, val_data = preprocess_data(dataset)

    # дальше уже trainer
    from core.trainer import Trainer
    trainer = Trainer()
    trainer.train(train_data, val_data)

if __name__ == "__main__":
    main()
