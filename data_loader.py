import yfinance as yf
import pandas as pd

def download_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)

    if df is None or df.empty:
        return pd.DataFrame()

    df.reset_index(inplace=True)

    # Flatten columns if MultiIndex (yfinance sometimes returns this)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    return df


# Optional test run (you can keep or remove)
if __name__ == "__main__":
    df = download_stock_data("AAPL", "2020-01-01", "2023-12-31")
    print(df.head())
    print("Shape:", df.shape)