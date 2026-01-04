# src/models/alpha_rag/rag_engine.py
import pandas as pd
import numpy as np
from .embeddings import TextEmbedder
from .llm_client import LLMClient

class FinancialRAG:
    def __init__(self):
        self.embedder = TextEmbedder()
        self.llm = LLMClient(provider="mock") # Cambiar a 'openai' si tienes key
        self.news_store = []
        self.vectors = None

    def ingest_data(self, df_news):
        """Loads a news DataFrame into the system."""
        if df_news.empty:
            print("⚠️ No news to ingest.")
            return
        
        # Save original texts
        self.news_store = df_news.to_dict('records')
        texts = [doc['title'] for doc in self.news_store]
        
        # Create vectors (The knowledge base)
        print(f"⚙️ Vectorizing {len(texts)} news items...")
        self.vectors = self.embedder.encode(texts)
        print("✅ Knowledge base updated.")

    def search(self, query, top_k=3):
        """Searches for news most semantically similar to the query."""
        if self.vectors is None or len(self.vectors) == 0:
            return []
            
        # Vectorize the query
        query_vec = self.embedder.encode([query])[0]
        
        # Cosine Similarity (Dot product because they are normalized)
        scores = np.dot(self.vectors, query_vec)
        
        # Get top_k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "score": float(scores[idx]),
                "title": self.news_store[idx]['title'],
                "date": self.news_store[idx]['date']
            })
        return results

    def get_trading_signal(self, ticker):
        """Full pipeline: Query -> Search -> LLM Analysis"""
        query = f"Is {ticker} stock outlook positive or negative based on recent events?"
        
        # 1. Retrieve relevant context (RAG)
        relevant_news = self.search(query, top_k=5)
        
        if not relevant_news:
            return "No data enough for signal."
            
        # 2. Format context for the LLM
        context_str = "\n".join([f"- {n['title']} (Score: {n['score']:.2f})" for n in relevant_news])
        
        # 3. Generate Insight
        print(f"\n🤖 [RAG] Context retrieved for {ticker}:")
        print(context_str)
        
        signal = self.llm.analyze_context(context_str, query)
        return signal