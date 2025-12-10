# src/data/ingest.py
import yfinance as yf
import pandas as pd
import numpy as np
import os
from statsmodels.tsa.stattools import adfuller

class MarketIngestor:
    def __init__(self, project_id, ticker='SPY', start='2010-01-01'):
        self.project_id = project_id
        self.ticker = ticker
        self.start = start
        self.data = None

    def fetch_data(self):
        """Downloads raw data from Yahoo Finance."""
        print(f"Fetching data for {self.ticker}...")
        self.data = yf.download(self.ticker, start=self.start)
        
        # Flatten MultiIndex columns if necessary (yfinance update fix)
        if isinstance(self.data.columns, pd.MultiIndex):
            self.data.columns = self.data.columns.get_level_values(0)
            
        return self.data

    def feature_engineering(self):
        """Calculates Log Returns and Realized Volatility."""
        # 1. Log Returns: r_t = ln(P_t / P_{t-1})
        self.data['log_ret'] = np.log(self.data['Adj Close'] / self.data['Adj Close'].shift(1))

        # 2. Target Variable: 21-day Realized Volatility (Annualized)
        # Formula: StdDev(returns) * sqrt(252 trading days)
        self.data['target_vol'] = self.data['log_ret'].rolling(window=21).std() * np.sqrt(252)

        # Drop NaNs created by shifting/rolling
        self.data.dropna(inplace=True)
        
        # Reset index so 'Date' becomes a column (needed for BigQuery)
        self.data.reset_index(inplace=True)
        return self.data

    def validate_stationarity(self):
        """Runs Augmented Dickey-Fuller (ADF) test on Returns."""
        print("\n--- Statistical Validation (ADF Test) ---")
        series = self.data['log_ret'].values
        result = adfuller(series)
        
        print(f"ADF Statistic: {result[0]}")
        print(f"p-value: {result[1]}")
        
        # Interpretation for the log
        if result[1] < 0.05:
            print("SUCCESS: Time Series is Stationary (p < 0.05). Ready for GARCH/LSTM.")
        else:
            print("WARNING: Time Series indicates Unit Root (Non-Stationary). Check differencing.")

    def save_to_bigquery(self):
        """Uploads the processed dataframe to BigQuery."""
        table_id = f"market_data.{self.ticker}_processed"
        print(f"\nUploading to BigQuery: {table_id}...")
        
        self.data.to_gbq(
            destination_table=table_id,
            project_id=self.project_id,
            if_exists='replace'  # Replaces table if it exists
        )
        print("Upload Complete.")

if __name__ == "__main__":
    # Get Project ID automatically from environment or hardcode it
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "YOUR_PROJECT_ID_HERE") 
    
    # Run Pipeline
    pipeline = MarketIngestor(project_id=PROJECT_ID)
    pipeline.fetch_data()
    pipeline.feature_engineering()
    pipeline.validate_stationarity()
    pipeline.save_to_bigquery()