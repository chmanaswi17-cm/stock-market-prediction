import joblib
import pandas as pd

model = joblib.load("saved_model.pkl")

sample = pd.DataFrame({
    'Open': [180],
    'High': [185],
    'Low': [178],
    'Volume': [50000000]
})

prediction = model.predict(sample)

print("Predicted Price:", prediction[0])