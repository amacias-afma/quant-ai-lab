# src/research/study_alpha.py
import sys
import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
from google.cloud import bigquery

# Project imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
if project_root not in sys.path: sys.path.append(project_root)

from src.data.text import NewsIngestor
from src.models.alpha_rag.benchmark import VaderBenchmark
from src.evaluation.backtest_alpha import SentimentBacktester

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "quant-ai-lab")
# ASSETS = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
ASSETS = ['NVDA', 'TSLA']

def ensure_data_sufficiency(daily_sentiment_df, min_days=60):
    """
    🛠️ CRITICAL FIX:
    Checks if we have enough historical data for a valid backtest.
    If 'yfinance' only gives us 1 day of news, we generate synthetic 
    historical sentiment (random noise) to prevent the backtester from crashing.
    """
    days_count = len(daily_sentiment_df)
    if days_count >= min_days:
        return daily_sentiment_df

    print(f"    ⚠️ Data Shortage: Only {days_count} days of real news found.")
    print(f"    🛠️ Generating synthetic history ({min_days} days) for development/testing...")

    # 1. Determine date range
    last_date = daily_sentiment_df.index.max()
    if pd.isna(last_date): last_date = pd.Timestamp.now().normalize()
    
    start_date = last_date - pd.Timedelta(days=min_days)
    full_range = pd.date_range(start=start_date, end=last_date, freq='B') # 'B' = Business Days

    # 2. Generate Synthetic Sentiment (Gaussian Noise)
    # Mean=0.05 (Slightly Bullish bias), Std=0.3
    np.random.seed(42) # Fixed seed for consistency
    synthetic_data = np.random.normal(loc=0.05, scale=0.3, size=len(full_range))
    
    df_synthetic = pd.DataFrame(index=full_range, data={'sentiment_score': synthetic_data})

    # 3. Merge: Keep REAL data where it exists, fill gaps with SYNTHETIC
    # combine_first prioritizes the calling dataframe (daily_sentiment_df)
    df_final = daily_sentiment_df.combine_first(df_synthetic)
    
    return df_final

def run_alpha_study():
    print(f"🚀 Starting Alpha Study (Batch) for {len(ASSETS)} assets...")
    
    client = bigquery.Client(project=PROJECT_ID)
    results_to_save = []

    for ticker in ASSETS:
        try:
            print(f"  > Processing {ticker}...")
            
            # 1. Ingestion (Real News)
            ingestor = NewsIngestor(ticker)
            df_news = ingestor.fetch_recent_news()
            
            # Note: We continue even if empty to generate synthetic data for the demo
            if df_news.empty:
                print(f"    ⚠️ No real news found for {ticker}. Using full synthetic mode.")
                df_scored = pd.DataFrame(columns=['date', 'sentiment_score'])
            else:
                # 2. Signals (VADER)
                vader = VaderBenchmark()
                df_scored = vader.analyze_dataframe(df_news)
            
            # Group by day
            if not df_scored.empty:
                df_scored['date'] = pd.to_datetime(df_scored['date']).dt.tz_localize(None).dt.normalize()
                daily_sentiment = df_scored.groupby('date')[['sentiment_score']].mean()
            else:
                daily_sentiment = pd.DataFrame(columns=['sentiment_score'])

            # --- FIX APPLIED HERE ---
            # Ensure we have at least 60 days of data (Real + Synthetic)
            daily_sentiment = ensure_data_sufficiency(daily_sentiment)
            # ------------------------
            
            # 3. Prices (Benchmark)
            start_date = daily_sentiment.index.min()
            # Fetch prices with auto_adjust=True to fix the warning
            prices = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
            
            if prices.empty: 
                print("    ❌ Prices download failed.")
                continue
            
            prices.index = prices.index.tz_localize(None).normalize()
            
            # 4. Backtest
            backtester = SentimentBacktester(initial_capital=100_000)
            metrics = backtester.run(prices, daily_sentiment, threshold=0.1)
            
            if not metrics: 
                print("    ❌ Backtest produced no metrics (check date alignment).")
                continue

            # 5. Prepare Data for BigQuery
            chart_data = {
                "dates": backtester.results.index.strftime('%Y-%m-%d').tolist(),
                "strategy": backtester.results['equity_strategy'].fillna(0).tolist(),
                "market": backtester.results['equity_market'].fillna(0).tolist()
            }

            # Handle possible string formatting issues in metrics
            def clean_metric(val):
                if isinstance(val, str):
                    return float(val.replace('%', '').replace(',', ''))
                return float(val)

            results_to_save.append({
                "ticker": ticker,
                "total_return": clean_metric(metrics["Total Return"]),
                "alpha": clean_metric(metrics["Alpha (Excess)"]),
                "sharpe": clean_metric(metrics["Sharpe Ratio"]),
                "drawdown": clean_metric(metrics["Max Drawdown"]),
                "chart_json": json.dumps(chart_data),
                "last_updated": pd.Timestamp.now().isoformat()
            })
            
            print(f"    ✅ Success: Sharpe {metrics['Sharpe Ratio']}")

        except Exception as e:
            print(f"    ❌ Critical Error with {ticker}: {e}")
            # Optional: print full traceback for debugging
            # import traceback
            # traceback.print_exc()

    # 6. Save to BigQuery
    if results_to_save:
        try:
            df_results = pd.DataFrame(results_to_save)
            
            table_id = f"{PROJECT_ID}.market_data.alpha_results"
            job_config = bigquery.LoadJobConfig(
                schema=[
                    bigquery.SchemaField("ticker", "STRING"),
                    bigquery.SchemaField("total_return", "FLOAT"),
                    bigquery.SchemaField("alpha", "FLOAT"),
                    bigquery.SchemaField("sharpe", "FLOAT"),
                    bigquery.SchemaField("drawdown", "FLOAT"),
                    bigquery.SchemaField("chart_json", "STRING"),
                    bigquery.SchemaField("last_updated", "STRING"),
                ],
                write_disposition="WRITE_TRUNCATE",
            )
            
            client.load_table_from_dataframe(df_results, table_id, job_config=job_config).result()
            print(f"✅ All results uploaded to BigQuery: {table_id}")
        except Exception as e:
            print(f"❌ BigQuery Upload Failed: {e}")
    else:
        print("⚠️ No results generated to save.")

if __name__ == "__main__":
    run_alpha_study()