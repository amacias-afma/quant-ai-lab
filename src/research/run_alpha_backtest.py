import sys
import os
import pandas as pd
import yfinance as yf

# Path adjustment for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
if project_root not in sys.path: sys.path.append(project_root)

from src.data.text import NewsIngestor
from src.models.alpha_rag.benchmark import VaderBenchmark
# If you want to test RAG (slow for historical backtest) uncomment:
# from src.models.alpha_rag.rag_engine import FinancialRAG 
from src.evaluation.backtest_alpha import SentimentBacktester

def run_study(ticker="TSLA"):
    print(f"--- 🔬 Starting Alpha Study for {ticker} ---")
    
    # 1. Get News (Text)
    # NOTE: yfinance free only gives recent news. 
    # For a real multi-year backtest, you would need a paid API (Polygon/Alpaca) or historical CSV.
    # Here we use available data to demonstrate the flow.
    ingestor = NewsIngestor(ticker)
    df_news = ingestor.fetch_recent_news()
    
    if df_news.empty:
        print("❌ Not enough news for backtest.")
        return

    # 2. Generate Signals (Modeling)
    print("🧠 Analyzing sentiment (VADER)...")
    model = VaderBenchmark()
    df_scored = model.analyze_dataframe(df_news)
    
    # Group sentiment by day (if multiple news same day)
    # Index must be Datetime without timezone to match yfinance
    df_scored['date'] = pd.to_datetime(df_scored['date']).dt.tz_localize(None).dt.normalize()
    daily_sentiment = df_scored.groupby('date')[['sentiment_score']].mean()
    
    # 3. Get Market Prices (Benchmark)
    print("📈 Downloading market prices...")
    # Search data from first found news date until today
    start_date = daily_sentiment.index.min()
    prices = yf.download(ticker, start=start_date, progress=False)
    prices.index = prices.index.tz_localize(None).normalize()
    
    # 4. Run Backtest
    backtester = SentimentBacktester(initial_capital=100000)
    metrics = backtester.run(prices, daily_sentiment, threshold=0.15)
    
    if metrics:
        print("\n📊 BACKTEST RESULTS:")
        for k, v in metrics.items():
            print(f"   {k}: {v}")
            
        # 5. Plot
        backtester.plot_results(ticker_name=ticker)
    else:
        print("⚠️ Could not execute backtest (date mismatch).")

if __name__ == "__main__":
    run_study()