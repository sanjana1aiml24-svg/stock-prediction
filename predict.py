from tensorflow.keras.models import load_model
from preprocess import prepare_data

import pandas as pd
import numpy as np

# Load trained model
model = load_model("stock_model.h5")

# Load saved stock data
df = pd.read_csv("stock_data.csv")

print(df.head())
print("Shape:", df.shape)

# Prepare data
X, y, scaler = prepare_data(df)

# Make predictions
predictions = model.predict(X)

# Convert predictions back to original scale
dummy_pred = np.zeros((len(predictions), 5))
dummy_pred[:, 3] = predictions.flatten()

predictions_actual = scaler.inverse_transform(dummy_pred)[:, 3]

# Convert actual values back
dummy_actual = np.zeros((len(y), 5))
dummy_actual[:, 3] = y.flatten()

actual_prices = scaler.inverse_transform(dummy_actual)[:, 3]

# Print prediction results
print("\nPredicted Prices:")
print(predictions_actual[:5])

print("\nActual Prices:")
print(actual_prices[:5])