import pandas as pd
import numpy as np
import yfinance as yf

def fetch_historical_data(tickers, start_date, end_date):
    """
    Downloads adjusted closing prices for a list of tickers from yfinance.
    """
    print(f"Fetching data for {tickers} from {start_date} to {end_date}...")
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data.dropna()

def calculate_log_returns(price_df):
    """
    Calculates daily logarithmic returns from price data.
    """
    return np.log(price_df / price_df.shift(1)).dropna()

if __name__ == "__main__":
    # Quick standalone test
    df = fetch_historical_data(['AAPL', 'MSFT', 'GOOG'], '2024-01-01', '2025-01-01')
    returns = calculate_log_returns(df)
    print(f"Successfully processed returns. Shape: {returns.shape}")
