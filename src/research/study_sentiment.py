# src/research/study_sentiment.py
import sys
import os
import pandas as pd

# Adjust path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
if project_root not in sys.path: sys.path.append(project_root)

from src.data.text import NewsIngestor
from src.models.alpha_rag.benchmark import VaderBenchmark

def run_initial_analysis(tickers=['NVDA', 'TSLA', 'AAPL']):
    print("🚀 Starting Sentiment Analysis (VADER)...")
    
    model = VaderBenchmark()
    all_results = []
    
    for ticker in tickers:
        # 1. Fetch News
        ingestor = NewsIngestor(ticker)
        df_news = ingestor.fetch_recent_news()
        
        if df_news.empty:
            continue
            
        # 2. Apply Model (Benchmark)
        df_scored = model.analyze_dataframe(df_news, text_column='title')
        
        # 3. Calculate "Sentiment Signal" average of the day
        avg_sentiment = df_scored['sentiment_score'].mean()
        print(f"   👉 {ticker}: Signal Promedio = {avg_sentiment:.4f}")
        
        all_results.append(df_scored)

    # Consolidate
    if all_results:
        final_df = pd.concat(all_results)
        # Show top 3 bullish and bearish news
        print("\n🏆 Top 3 Bullish News:")
        print(final_df.nlargest(3, 'sentiment_score')[['ticker', 'title', 'sentiment_score']])
        
        print("\n💀 Top 3 Bearish News:")
        print(final_df.nsmallest(3, 'sentiment_score')[['ticker', 'title', 'sentiment_score']])
        
        return final_df
    else:
        print("No data found.")
        return None

if __name__ == "__main__":
    run_initial_analysis()