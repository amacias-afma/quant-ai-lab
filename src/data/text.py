# src/data/text.py
import yfinance as yf
import pandas as pd
from datetime import datetime

class NewsIngestor:
    def __init__(self, ticker):
        self.ticker = ticker
    
    def fetch_recent_news(self):
        """
        Downloads recent news using the public yfinance API.
        Returns a DataFrame with: [date, title, publisher, link]
        """
        try:
            stock = yf.Ticker(self.ticker)
            news = stock.news
            
            data = []
            # print(news)
            for item in news:

                # Convert timestamp to readable date
                # Parse ISO date string (e.g., '2025-12-30T11:02:12Z')
                pub_date = datetime.fromisoformat(item['content']['pubDate'].replace('Z', '+00:00'))
                data.append({
                    "date": pub_date,
                    "ticker": self.ticker,
                    "title": item['content']['title'],
                    "publisher": item['content']['provider']['displayName'],
                    "link": item['content']['canonicalUrl']['url']
                })
            
            df = pd.DataFrame(data)
            print(f"📰 {len(df)} news found for {self.ticker}")
            return df
            
        except Exception as e:
            print(f"⚠️ Error fetching news for {self.ticker}: {e}")
            return pd.DataFrame()

# Fast Test
if __name__ == "__main__":
    ingestor = NewsIngestor("NVDA")
    print(ingestor.fetch_recent_news().head())