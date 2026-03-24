import numpy as np
from data_loader import download_stock_data
from preprocess import prepare_data
from model import build_model
from sklearn.model_selection import train_test_split


# Load data
df = download_stock_data("AAPL", "2020-01-01", "2023-12-31")

# Preprocess
X, y, scaler = prepare_data(df)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Build model
model = build_model((X.shape[1], X.shape[2]))

# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Evaluate
loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)

# Save model
model.save("stock_model.h5")
print("Model saved successfully")