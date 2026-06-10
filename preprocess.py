import pandas as pd
from pathlib import Path

csv_path = Path(__file__).with_name("AAPL.csv") / "HistoricalQuotes.csv"
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

close_column = "Close/Last" if "Close/Last" in df.columns else "Close"
df[close_column] = df[close_column].replace(r'[$,]', '', regex=True).astype(float)

df['Prediction'] = df[close_column].shift(-1)

print(df[[close_column, 'Prediction']].head())