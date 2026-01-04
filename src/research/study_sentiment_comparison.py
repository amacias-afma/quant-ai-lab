# src/research/study_sentiment_comparison.py
# ... (system imports for path) ...
from src.data.text import NewsIngestor
from src.models.alpha_rag.benchmark import VaderBenchmark
from src.models.alpha_rag.rag_engine import FinancialRAG

def compare_models(ticker="TSLA"):
    # 1. Get Data
    ingestor = NewsIngestor(ticker)
    df = ingestor.fetch_recent_news()
    
    # 2. Model 1: VADER (Benchmark)
    print("\n--- MODEL 1: VADER (Lexical) ---")
    vader = VaderBenchmark()
    df_vader = vader.analyze_dataframe(df)
    vader_score = df_vader['sentiment_score'].mean()
    print(f"Average VADER Score: {vader_score:.4f}")
    
    # 3. Model 2: RAG (Semantic)
    print("\n--- MODEL 2: RAG (Semantic) ---")
    rag = FinancialRAG()
    rag.ingest_data(df)
    
    # Specific semantic search (e.g., Supply chain risks)
    insight = rag.get_trading_signal(ticker)
    print(f"RAG Insight: {insight}")

if __name__ == "__main__":
    compare_models()