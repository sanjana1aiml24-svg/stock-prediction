from data_loader import download_stock_data

ticker = "AAPL"
start_date = "2020-01-01"
end_date = "2023-12-31"

df = download_stock_data(ticker, start_date, end_date)

if df.empty:
    print("Data download failed. Try again.")
else:
    print(df.head())
    print("Shape:", df.shape)