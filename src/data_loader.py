import pandas as pd
import numpy as np
import yfinance as yf

def fetch_historical_data(tickers, start_date, end_date):
    """
    Downloads historical adjusted closing prices for a list of tickers from yfinance.
    Uses auto_adjust=False to safely preserve the explicit 'Adj Close' column.
    """
    print(f"Fetching data for {tickers} from {start_date} to {end_date}...")
    
    # Force auto_adjust=False to guarantee 'Adj Close' is present in the download
    raw_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
    
    # Extract only the 'Adj Close' metrics level
    data = raw_data['Adj Close']
    
    # Clean up standalone Series formatting into single DataFrames if needed
    if isinstance(data, pd.Series):
        data = data.to_frame()
        
    return data.dropna()

def calculate_log_returns(price_df):
    """
    Calculates daily logarithmic returns from price data.
    """
    return np.log(price_df / price_df.shift(1)).dropna()

if __name__ == "__main__":
    # Test script locally
    df = fetch_historical_data(['AAPL', 'MSFT', 'GOOG'], '2024-01-01', '2025-01-01')
    returns = calculate_log_returns(df)
    print("\n--- Download Sample Head ---")
    print(df.head())
    print(f"\nSuccessfully processed returns. Data Shape: {returns.shape}")
