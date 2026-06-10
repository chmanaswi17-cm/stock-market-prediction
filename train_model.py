import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

csv_path = Path(__file__).with_name("AAPL.csv") / "HistoricalQuotes.csv"
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

for column in ["Close/Last", "Open", "High", "Low"]:
    df[column] = df[column].replace(r'[$,]', '', regex=True).astype(float)

df['Volume'] = df['Volume'].astype(int)

df['Prediction'] = df['Close/Last'].shift(-1)

df = df[:-1]

X = df[['Open', 'High', 'Low', 'Volume']]

y = df['Prediction']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

print("Model Trained Successfully")
predictions = model.predict(X_test)

print(predictions[:5])
from sklearn.metrics import r2_score

score = r2_score(y_test, predictions)

print("Accuracy:", score)
import joblib

joblib.dump(model, "saved_model.pkl")

print("Model Saved")