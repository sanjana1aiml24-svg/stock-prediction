import numpy as np
from sklearn.preprocessing import MinMaxScaler


def prepare_data(df, time_steps=30):
    """
    Convert stock data into sequences for LSTM.
    """

    # Use only numeric features
    data = df[['Open', 'High', 'Low', 'Close', 'Volume']].values

    # Normalize data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X = []
    y = []

    # Create sequences
    for i in range(time_steps, len(data_scaled)):
        X.append(data_scaled[i - time_steps:i])
        y.append(data_scaled[i, 3])  # Close price

    X = np.array(X)
    y = np.array(y)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    return X, y, scaler