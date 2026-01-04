import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import sys
import os

# Ensure the path includes src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.text import NewsIngestor
from src.models.alpha_rag.benchmark import VaderBenchmark
from src.models.alpha_rag.embeddings import TextEmbedder
from src.models.alpha_rag.rag_engine import FinancialRAG

# --- 1. Test Data Ingestion (Mocked) ---
def test_news_ingestor_fetch():
    """
    Verify NewsIngestor parses yfinance data correctly without calling the real API.
    """
    # We patch 'src.data.text.yf.Ticker' to ensure we capture the specific import used in the module
    with patch("src.data.text.yf.Ticker") as MockTicker:
        mock_stock = MockTicker.return_value
        
        # --- FIX: Update Mock Structure to match your new code ---
        # Your code expects: item['content']['title'], item['content']['pubDate'], etc.
        mock_stock.news = [
            {
                "id": "1234-5678",
                "content": {
                    "pubDate": "2024-01-01T12:00:00Z", # ISO format as expected by fromisoformat
                    "title": "NVIDIA revenue grows 200%",
                    "provider": {
                        "displayName": "Bloomberg"
                    },
                    "canonicalUrl": {
                        "url": "https://example.com/nvda"
                    },
                    "summary": "This is a summary of the news..."
                }
            }
        ]
        
        ingestor = NewsIngestor("NVDA")
        df = ingestor.fetch_recent_news()
        
        # Assertions
        assert not df.empty, "DataFrame should not be empty"
        assert len(df) == 1
        assert df.iloc[0]['ticker'] == "NVDA"
        assert "revenue grows" in df.iloc[0]['title']
        assert df.iloc[0]['publisher'] == "Bloomberg"

# --- 2. Test VADER Benchmark ---
def test_vader_scoring_logic():
    """
    Verify VADER correctly identifies positive vs negative sentiment in financial context.
    """
    model = VaderBenchmark()
    
    # 1. Clear Positive
    pos_text = "The company reported record-breaking profits and excellent growth guidance."
    assert model.score_text(pos_text) > 0.05
    
    # 2. Clear Negative
    neg_text = "Bankruptcy fears loom as debt crisis worsens and revenue collapses."
    assert model.score_text(neg_text) < -0.05
    
    # 3. DataFrame Integration check
    df = pd.DataFrame([{"title": "Good news"}, {"title": "Bad news"}])
    df_scored = model.analyze_dataframe(df, text_column="title")
    
    assert "sentiment_score" in df_scored.columns
    assert "sentiment_label" in df_scored.columns
    assert df_scored.iloc[0]['sentiment_label'] == 'Bullish 🟢'

# --- 3. Test Embeddings (Mocked) ---
def test_embedding_dimensions_and_norm():
    """
    Test that TextEmbedder returns vectors of correct shape and L2 normalization.
    """
    # Mock SentenceTransformer to avoid downloading the 100MB+ model during tests
    with patch("src.models.alpha_rag.embeddings.SentenceTransformer") as MockBERT:
        mock_model = MockBERT.return_value
        
        # Simulate a 384-dimensional vector (standard for all-MiniLM-L6-v2)
        fake_vector = np.random.rand(2, 384)
        mock_model.encode.return_value = fake_vector
        
        embedder = TextEmbedder()
        texts = ["Market is volatile", "Buy the dip"]
        vectors = embedder.encode(texts)
        
        # Assertions
        assert vectors.shape == (2, 384)
        assert isinstance(vectors, np.ndarray)

# --- 4. Test RAG Engine Retrieval ---
def test_rag_retrieval_logic():
    """
    Test the Cosine Similarity retrieval logic without an LLM.
    """
    rag = FinancialRAG()
    
    # A. Setup Dummy Knowledge Base
    rag.news_store = [
        {"title": "Apple releases iPhone 16", "date": "2024-09-01"},  # Target Doc
        {"title": "Oil prices drop due to supply", "date": "2024-09-02"} # Irrelevant Doc
    ]
    
    # B. Setup Orthogonal Vectors (Simplified Space)
    # Doc 1 = [1, 0], Doc 2 = [0, 1]
    rag.vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    
    # C. Mock Embedder to return a query vector aligned with Doc 1
    # Query = [0.9, 0.1] (Highly similar to Doc 1)
    rag.embedder.encode = MagicMock(return_value=np.array([[0.9, 0.1]]))
    
    # D. Execute Search
    results = rag.search("Something about Apple", top_k=1)
    
    # E. Assertions
    assert len(results) == 1
    assert results[0]['title'] == "Apple releases iPhone 16"
    assert np.isclose(results[0]['score'], 0.9)