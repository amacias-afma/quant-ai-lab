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

import yfinance as yf
import pandas as pd
import numpy as np
import os
from google.cloud import bigquery

# The "Volatile 10" Configuration
TICKER_CONFIG = {
    "^GSPC": "S&P 500",
    "BTC-USD": "Bitcoin",
    "CLP=X": "USD/CLP (Chile Peso)",
    "SQM": "SQM (Lithium)",
    "HG=F": "Copper Futures",
    "TSLA": "Tesla",
    "NVDA": "NVIDIA",
    "CL=F": "Crude Oil",
    "TLT": "US Treasuries (20Y)",
    "VXX": "VIX Volatility"
}

class BatchMarketIngestor:
    def __init__(self, project_id):
        self.project_id = project_id
        self.client = bigquery.Client(project=project_id)
        # Create dataset if not exists
        self.dataset_id = f"{self.project_id}.market_data"
        self.client.create_dataset(self.dataset_id, exists_ok=True)

    def fetch_and_process(self, ticker):
        """Fetches data, calculates returns/variance, returns DataFrame."""
        print(f"📉 Processing {ticker}...")
        
        # Fetch 10 years of data
        # Note: yfinance auto-adjusts for splits/dividends
        df = yf.download(ticker, start="2015-01-01", progress=False)
        
        if df.empty:
            print(f"⚠️ Warning: No data found for {ticker}")
            return None

        # Handle MultiIndex columns (yfinance update fix)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 1. Feature Engineering
        # Log Returns: r_t = ln(P_t / P_{t-1})
        df['log_ret'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
        
        # Target Variance: r_{t+1}^2 (Shifted back by 1)
        # We model today's return variance using yesterday's info
        df['target_variance'] = df['log_ret'] ** 2
        
        # IMPORTANT: Forward Fill for "Next Day" Prediction alignment
        # The model at time T predicts T+1. 
        # So we align: Features(T) -> Target(T+1)
        df['next_day_variance'] = df['target_variance'].shift(-1)
        
        # 2. Add Metadata
        df['ticker'] = ticker
        df['asset_name'] = TICKER_CONFIG.get(ticker, "Unknown")
        
        df.dropna(inplace=True)
        df.reset_index(inplace=True)
        
        # Ensure column names are BigQuery friendly (lowercase, no spaces)
        df.rename(columns={'Date': 'date', 'Adj Close': 'price'}, inplace=True)
        
        # Select only necessary columns to save space/cost
        final_df = df[['date', 'ticker', 'asset_name', 'price', 'log_ret', 'target_variance', 'next_day_variance']]
        
        return final_df

    def save_to_bigquery(self, df):
        """Appends data to a single Master Table."""
        table_id = f"{self.dataset_id}.historical_prices"
        
        job_config = bigquery.LoadJobConfig(
            # Append to existing table so we hold all tickers in one place
            write_disposition="WRITE_APPEND", 
            schema=[
                bigquery.SchemaField("date", "TIMESTAMP"),
                bigquery.SchemaField("ticker", "STRING"),
                bigquery.SchemaField("asset_name", "STRING"),
                bigquery.SchemaField("price", "FLOAT"),
                bigquery.SchemaField("log_ret", "FLOAT"),
                bigquery.SchemaField("target_variance", "FLOAT"),
                bigquery.SchemaField("next_day_variance", "FLOAT"),
            ],
            time_partitioning=bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field="date"  # Partition by date for cheaper queries
            )
        )
        
        print(f"🚀 Uploading {len(df)} rows to {table_id}...")
        job = self.client.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result()  # Wait for job to complete
        print("Done.")

    def run_pipeline(self):
        # Clean the table first (Full Refresh)
        table_id = f"{self.dataset_id}.historical_prices"
        self.client.delete_table(table_id, not_found_ok=True)
        print("🧹 Cleaned old BigQuery table.")

        for ticker in TICKER_CONFIG.keys():
            df = self.fetch_and_process(ticker)
            if df is not None:
                self.save_to_bigquery(df)

if __name__ == "__main__":
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "quant-ai-lab")
    # For local test: PROJECT_ID = "your-project-id"
    
    ingestor = BatchMarketIngestor(PROJECT_ID)
    ingestor.run_pipeline()