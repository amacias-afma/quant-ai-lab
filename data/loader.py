# src/data/loader.py
import yfinance as yf
import pandas as pd
import numpy as np

class MarketDataLoader:
    """
    Fetches and processes market data for volatility modeling.
    """
    def __init__(self, ticker: str, start_date: str, end_date: str):
        self.ticker = ticker
        self.start = start_date
        self.end = end_date
        self.data = None

    def fetch_data(self):
        """Fetches OHL data from Yahoo Finance."""
        print(f"Fetching data for {self.ticker}...")
        self.data = yf.download(self.ticker, start=self.start, end=self.end)
        return self.data

    def calculate_log_returns(self):
        """Calculates Log Returns: r_t = ln(P_t / P_{t-1})"""
        if self.data is None:
            raise ValueError("Data not fetched. Call fetch_data() first.")
        
        # Using 'Adj Close' for total return
        self.data['Log_Ret'] = np.log(self.data[('Close', self.ticker)] / self.data[('Close', self.ticker)].shift(1))
        
        # Calculate a naive Realized Volatility (21-day rolling std dev) as a target
        self.data['Realized_Vol'] = self.data['Log_Ret'].rolling(window=21).std() * np.sqrt(252)
        
        self.data.dropna(inplace=True)
        return self.data[['Log_Ret', 'Realized_Vol']]

if __name__ == "__main__":
    loader = MarketDataLoader('SPY', '2015-01-01', '2025-01-01')
    df = loader.fetch_data()
    processed_df = loader.calculate_log_returns()
    print(processed_df.head())
    # Next Step: Save to BigQuery or CSV