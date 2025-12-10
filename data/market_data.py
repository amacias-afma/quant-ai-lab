import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
# from google.cloud import bigquery
import os

class MarketData:

    def __init__(self, ticker: str, start_date: str, end_date: str=None):
        self.ticker = ticker
        self.start = start_date
        self.end = end_date
        self.data = None
    
    def fetch_data_yfinance(self):
        """Downloads raw data from Yahoo Finance."""
        print(f"Fetching data for {self.ticker}...")
        self.data = yf.download(self.ticker, start=self.start, end=self.end)
        
        # Flatten MultiIndex columns if necessary (yfinance update fix)
        if isinstance(self.data.columns, pd.MultiIndex):
            self.data.columns = self.data.columns.get_level_values(0)
        self.data = self.data[['Close']]
        self.data.columns = ['price']
        return self.data
    
    def feature_engineering(self, window: int=21, trading_days: int=252):
        """Calculates Log Returns and Realized Volatility."""
        # 1. Log Returns: r_t = ln(P_t / P_{t-1})
        self.data['log_ret'] = np.log(self.data['price'] / self.data['price'].shift(1))
        
        # 2. The "True" Target for ML Training (Squared Returns)
        # We want to predict Volatility at T+1 using data up to T.
        # So the target for row T is r_{T+1}^2
        # We calculate it now, and shift it so row T contains the future value.
        self.data['target_variance'] = self.data['log_ret'] ** 2
        
        # Shift -1: The value at index 'i' now holds the return for 'i+1'
        # This is what we want the model to predict given features at 'i'.
        self.data['next_day_variance'] = self.data['target_variance'].shift(-1)

        # 3. Auxiliary Target: 21-Day Realized Vol (Only for visual smoothing/reference)
        # This is useful for plotting, even if we don't optimize on it.
        self.data['realized_vol_21d'] = self.data['log_ret'].rolling(window=window).std() * np.sqrt(trading_days)

        # Drop NaNs created by shifting/rolling
        self.data.dropna(inplace=True)
        
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
            self.stationary = True
        else:
            print("WARNING: Time Series indicates Unit Root (Non-Stationary). Check differencing.")
            self.stationary = False
        
        return result
    
    def save_to_bigquery(self):
        """Uploads the processed dataframe to BigQuery."""
        self.project_id = os.getenv('PROJECT_ID', 'quant-ai-lab')
        table_id = f"market_data.{self.ticker}_processed"
        print(f"\nUploading to BigQuery: {table_id}...")
        # Reset index so 'Date' becomes a column (needed for BigQuery)
        self.data.reset_index(inplace=True)
        self.data.columns = self.data.columns.str.lower()
        self.data.to_gbq(
            destination_table=table_id,
            project_id=self.project_id,
            if_exists='replace'  # Replaces table if it exists
        )
        print("Upload Complete.")
    
    def save_to_parquet(self, path: str=None):
        """Saves the processed dataframe to a Parquet file."""
        file_name = f"{self.ticker}_processed.parquet"
        if path is None:
            path = file_name
        else:
            path = os.path.join(path, file_name)

        # Clear metadata attributes which can cause pyarrow serialization errors
        self.data.attrs = {}

        # Ensure Date is timezone-naive to avoid compatibility issues
        if 'date' in self.data.columns and pd.api.types.is_datetime64_any_dtype(self.data['date']):
            try:
                self.data['date'] = self.data['date'].dt.tz_localize(None)
            except Exception:
                pass # Already naive or other issue

        self.data.to_parquet(path, index=True)
        print(f"Parquet saved successfully to {path}")
    
    def run_pipeline(self):
        """Runs the entire pipeline."""
        self.fetch_data_yfinance()
        self.feature_engineering()
        self.validate_stationarity()
        self.save_to_bigquery()
