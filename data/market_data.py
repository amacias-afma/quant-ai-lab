# import yfinance as yf
# import pandas as pd
import pandas_gbq
# import numpy as np
# from statsmodels.tsa.stattools import adfuller
# # from google.cloud import bigquery
import os
from scipy.stats import norm

class MarketData:

    def __init__(self, ticker: str, start_date: str=None, end_date: str=None, project_id: str=None):
        self.ticker = ticker
        self.start = start_date
        self.end = end_date
        self.data = None
        self.models = []
        self.project_id = os.getenv('PROJECT_ID', project_id)
    
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
            df.sort_index(inplace=True)
            # df = df[(df.index >= self.start) & (df.index <= self.end)]
            df = df[['price']]

            self.data = df
            # return df
        elif source == 'local':
            df = pd.read_parquet(f"data/{self.ticker}_processed.parquet")
            df.set_index('date', inplace=True)
            self.data = df
            # return df
        else:
            raise ValueError("Invalid source. Must be 'bigquery' or 'local'.")
    
    def replace_outliers(self, alpha=0.0001):
        min_outlier = self.data.price.mean() + self.data.price.std() * norm.ppf(alpha)
        max_outlier = self.data.price.mean() + self.data.price.std() * norm.ppf(1-alpha)

        # Remove outliers
        df_outlier = self.data.loc[(self.data.price < min_outlier) | (self.data.price > max_outlier), 'price'].copy()
        print("Outliers:")
        print(df_outlier)

        self.data.loc[(self.data.price < min_outlier) | (self.data.price > max_outlier), 'price'] = np.nan
        self.data.ffill(inplace=True)

    def save_to_parquet(self):
        self.data.to_parquet(f"data/{self.ticker}_processed.parquet")

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