from pathlib import Path

import pandas as pd

data_dir = Path(__file__).resolve().parent / "data"
orders_path = data_dir / "olist_orders_dataset.csv"

if not orders_path.exists():
	matches = list(data_dir.glob("olist_orders_dataset.csv*"))
	if matches:
		orders_path = matches[0]

orders = pd.read_csv(orders_path)

print("DATASET INFO")
print(orders.info())

print("\nMISSING VALUES")
print(orders.isnull().sum())