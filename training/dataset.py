import pandas as pd
from datasets import Dataset

def load_dataset(path: str) -> Dataset:
    """Загрузка датасета из CSV и преобразование в HuggingFace Dataset"""
    df = pd.read_csv(path)
    
    # Убедимся, что есть колонка "text"
    if "text" not in df.columns:
        raise ValueError("❌ В CSV должен быть столбец 'text'")
    
    dataset = Dataset.from_pandas(df)
    return dataset

if __name__ == "__main__":
    dataset = load_dataset("data/train.csv")
    
    # Сохраняем в формате HuggingFace (как требует run_train.py)
    dataset.save_to_disk("data/cleaned/")
    print("✅ Датасет сохранён в data/cleaned/")
    print(dataset)
