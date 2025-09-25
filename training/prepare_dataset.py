from datasets import Dataset, DatasetDict
import pandas as pd
import os


raw_csv = "data/raw/data.csv"  
df = pd.read_csv(raw_csv)
ds = Dataset.from_pandas(df[['text']].dropna())
ds = ds.train_test_split(test_size=0.1)
os.makedirs("data/cleaned", exist_ok=True)
ds.save_to_disk("data/cleaned")
print("Saved dataset to data/cleaned/")
