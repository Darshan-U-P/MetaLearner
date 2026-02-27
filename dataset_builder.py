import pandas as pd
import os


os.makedirs("datasets", exist_ok=True)

df = pd.read_json("logs/meta_logs.jsonl", lines=True)

df.to_csv("datasets/meta_dataset.csv", index=False)

print("Dataset created at datasets/meta_dataset.csv")