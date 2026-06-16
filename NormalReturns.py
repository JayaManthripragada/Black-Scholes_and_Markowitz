import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm


def download_data(stock, start_date, end_date):
    ticker = yf.download(stock, start=start_date, end=end_date, auto_adjust=True)
    if ticker.empty:
        raise ValueError("Downloaded data is empty. Check ticker or date range.")
    return ticker[['Close']].rename(columns={'Close': 'Price'})



def calculate_returns(stock_data):
    stock_data = np.log(stock_data['Price'] / stock_data['Price'].shift(1))
    return stock_data[1:]


def show_plot(stock_data):
    plt.hist(stock_data, bins=300)
    stock_variance = stock_data.var()
    stock_mean = stock_data.mean()
    sigma = np.sqrt(stock_variance)
    x = np.linspace(stock_mean - 3 * sigma, stock_mean + 3 * sigma, 100)
    plt.plot(x, norm.pdf(x, stock_mean, sigma))
    plt.show()


if __name__ == "__main__":
    stock = download_data('^GSPC', '2015-01-01', '2025-01-01')
    #print(stock)
    log_daily_returns = calculate_returns(stock)
    show_plot(log_daily_returns)
