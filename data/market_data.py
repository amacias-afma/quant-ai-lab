# import yfinance as yf
# import pandas as pd
import pandas_gbq
# import numpy as np
# from statsmodels.tsa.stattools import adfuller
# # from google.cloud import bigquery
import os

class MarketData:

    def __init__(self, ticker: str, start_date: str=None, end_date: str=None, project_id: str=None):
        self.ticker = ticker
        self.start = start_date
        self.end = end_date
        self.data = None
        self.project_id = os.getenv('PROJECT_ID', project_id)


#     def fetch_data_yfinance(self):
#         """Downloads raw data from Yahoo Finance."""
#         print(f"Fetching data for {self.ticker}...")
#         self.data = yf.download(self.ticker, start=self.start, end=self.end)
        
#         # Flatten MultiIndex columns if necessary (yfinance update fix)
#         if isinstance(self.data.columns, pd.MultiIndex):
#             self.data.columns = self.data.columns.get_level_values(0)
#         self.data = self.data[['Close']]
#         self.data.columns = ['price']
#         return self.data
    
#     def feature_engineering(self, window: int=21, trading_days: int=252):
#         """Calculates Log Returns and Realized Volatility."""
#         # 1. Log Returns: r_t = ln(P_t / P_{t-1})
#         self.data['log_ret'] = np.log(self.data['price'] / self.data['price'].shift(1))
        
#         # 2. The "True" Target for ML Training (Squared Returns)
#         # We want to predict Volatility at T+1 using data up to T.
#         # So the target for row T is r_{T+1}^2
#         # We calculate it now, and shift it so row T contains the future value.
#         self.data['target_variance'] = self.data['log_ret'] ** 2
        
#         # Shift -1: The value at index 'i' now holds the return for 'i+1'
#         # This is what we want the model to predict given features at 'i'.
#         self.data['next_day_variance'] = self.data['target_variance'].shift(-1)

#         # 3. Auxiliary Target: 21-Day Realized Vol (Only for visual smoothing/reference)
#         # This is useful for plotting, even if we don't optimize on it.
#         self.data['realized_vol_21d'] = self.data['log_ret'].rolling(window=window).std() * np.sqrt(trading_days)

#         # Drop NaNs created by shifting/rolling
#         self.data.dropna(inplace=True)
        
#         return self.data
    
    def load_data(self, source='bigquery'):
        """Fetches training data from BigQuery."""
        if source == 'bigquery':
            query = f"""
                SELECT * 
                FROM `{self.project_id}.market_data.historical_prices`
                WHERE ticker = '{self.ticker}'
                ORDER BY date ASC
            """
            print("Fetching data from BigQuery...")
            df = pandas_gbq.read_gbq(query, project_id=self.project_id)
            df.set_index('date', inplace=True)
            return df
        elif source == 'local':
            df = pd.read_parquet(f"data/{self.ticker}_processed.parquet")
            df.set_index('date', inplace=True)
            return df
        else:
            raise ValueError("Invalid source. Must be 'bigquery' or 'local'.")

#     def validate_stationarity(self):
#         """Runs Augmented Dickey-Fuller (ADF) test on Returns."""
#         print("\n--- Statistical Validation (ADF Test) ---")
#         series = self.data['log_ret'].values
#         result = adfuller(series)
        
#         print(f"ADF Statistic: {result[0]}")
#         print(f"p-value: {result[1]}")
        
#         # Interpretation for the log
#         if result[1] < 0.05:
#             print("SUCCESS: Time Series is Stationary (p < 0.05). Ready for GARCH/LSTM.")
#             self.stationary = True
#         else:
#             print("WARNING: Time Series indicates Unit Root (Non-Stationary). Check differencing.")
#             self.stationary = False
        
#         return result
    
#     def save_to_bigquery(self, project_id: str=None):
#         """Uploads the processed dataframe to BigQuery."""
#         self.project_id = os.getenv('PROJECT_ID', project_id)
#         table_id = f"market_data.{self.ticker}_processed"
#         print(f"\nUploading to BigQuery: {table_id}...")
#         # Reset index so 'Date' becomes a column (needed for BigQuery)
#         self.data.reset_index(inplace=True)
#         self.data.columns = self.data.columns.str.lower()
#         pandas_gbq.to_gbq(
#             self.data,
#             destination_table=table_id,
#             project_id=self.project_id,
#             if_exists='replace'  # Replaces table if it exists
#         )
#         print("Upload Complete.")
    
#     def save_to_parquet(self, path: str=None):
#         """Saves the processed dataframe to a Parquet file."""
#         file_name = f"{self.ticker}_processed.parquet"
#         if path is None:
#             path = file_name
#         else:
#             path = os.path.join(path, file_name)

#         # Clear metadata attributes which can cause pyarrow serialization errors
#         self.data.attrs = {}

#         # Ensure Date is timezone-naive to avoid compatibility issues
#         if 'date' in self.data.columns and pd.api.types.is_datetime64_any_dtype(self.data['date']):
#             try:
#                 self.data['date'] = self.data['date'].dt.tz_localize(None)
#             except Exception:
#                 pass # Already naive or other issue

#         self.data.to_parquet(path, index=True)
#         print(f"Parquet saved successfully to {path}")
    
#     def run_pipeline(self):
#         """Runs the entire pipeline."""
#         self.fetch_data_yfinance()
#         self.feature_engineering()
#         self.validate_stationarity()
#         self.save_to_bigquery()


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

    # def feature_engineering(self, window: int=21, trading_days: int=252):
    #     """Calculates Log Returns and Realized Volatility."""
    #     # 1. Log Returns: r_t = ln(P_t / P_{t-1})
    #     self.data['log_ret'] = np.log(self.data['price'] / self.data['price'].shift(1))
        
    #     # 2. The "True" Target for ML Training (Squared Returns)
    #     # We want to predict Volatility at T+1 using data up to T.
    #     # So the target for row T is r_{T+1}^2
    #     # We calculate it now, and shift it so row T contains the future value.
    #     self.data['target_variance'] = self.data['log_ret'] ** 2
        
    #     # Shift -1: The value at index 'i' now holds the return for 'i+1'
    #     # This is what we want the model to predict given features at 'i'.
    #     self.data['next_day_variance'] = self.data['target_variance'].shift(-1)

    #     # 3. Auxiliary Target: 21-Day Realized Vol (Only for visual smoothing/reference)
    #     # This is useful for plotting, even if we don't optimize on it.
    #     self.data['realized_vol_21d'] = self.data['log_ret'].rolling(window=window).std() * np.sqrt(trading_days)

    #     # Drop NaNs created by shifting/rolling
    #     self.data.dropna(inplace=True)
        
    #     return self.data
    

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
            df = df[['Close']]            
        
        # Log Returns: r_t = ln(P_t / P_{t-1})
        df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Target Variance: r_{t+1}^2 (Shifted back by 1)
        # We model today's return variance using yesterday's info
        df['target_variance'] = df['log_ret'] ** 2
        
        # IMPORTANT: Forward Fill for "Next Day" Prediction alignment
        # The model at time T predicts T+1. 
        # So we align: Features(T) -> Target(T+1)
        df['next_day_variance'] = df['target_variance'].shift(-1)

        # 3. Auxiliary Target: 21-Day Realized Vol (Only for visual smoothing/reference)
        # This is useful for plotting, even if we don't optimize on it.
        df['realized_vol_21d'] = df['log_ret'].rolling(window=21).std() * np.sqrt(252)
        
        # 2. Add Metadata
        df['ticker'] = ticker
        df['asset_name'] = TICKER_CONFIG.get(ticker, "Unknown")
        
        df.dropna(inplace=True)
        df.reset_index(inplace=True)
        
        # Ensure column names are BigQuery friendly (lowercase, no spaces)
        df.rename(columns={'Date': 'date', 'Close': 'price'}, inplace=True)
        
        # Select only necessary columns to save space/cost
        final_df = df[['date', 'ticker', 'asset_name', 'price', 'log_ret', 'target_variance', 'next_day_variance', 'realized_vol_21d']]
        
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