import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from tensorflow.keras.models import load_model
from preprocess import prepare_data

# Load trained model
model = load_model("stock_model.h5")

# Page title
st.title("📈 AI Stock Market Predictor")

# Load stock data
df = pd.read_csv("stock_data.csv")

# Show latest stock info
latest_close = df['Close'].iloc[-1]

st.subheader("Current Stock Price")
st.write(f"${latest_close:.2f}")

# Prepare data
X, y, scaler = prepare_data(df)

# Predict
predictions = model.predict(X)

# Convert predictions back
dummy_pred = np.zeros((len(predictions), 5))
dummy_pred[:, 3] = predictions.flatten()

predicted_prices = scaler.inverse_transform(dummy_pred)[:, 3]

# Latest prediction
predicted_next = predicted_prices[-1]

st.subheader("Predicted Next Close Price")
st.write(f"${predicted_next:.2f}")

# Candlestick chart
fig = go.Figure(data=[go.Candlestick(
    x=df['Date'],
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close']
)])

fig.update_layout(
    title="Stock Market Chart",
    xaxis_title="Date",
    yaxis_title="Price"
)

st.plotly_chart(fig)

# Prediction graph
fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    y=predicted_prices,
    mode='lines',
    name='Predicted'
))

actual_close = df['Close'].values[30:]

fig2.add_trace(go.Scatter(
    y=actual_close,
    mode='lines',
    name='Actual'
))

fig2.update_layout(
    title="Actual vs Predicted Prices",
    xaxis_title="Time",
    yaxis_title="Price"
)

st.plotly_chart(fig2)