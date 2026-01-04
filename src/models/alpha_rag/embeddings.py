# src/models/alpha_rag/embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np

class TextEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Lightweight and very fast model, standard in the industry for simple RAG
        print(f"🧠 Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        """Convert a list of texts into a matrix of vectors."""
        if not texts:
            return np.array([])
        # Convert to vectors and normalize for cosine search
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        # L2 normalization for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms