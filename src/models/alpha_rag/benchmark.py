# src/models/alpha_rag/benchmark.py
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

# Download the lexicon the first time
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class VaderBenchmark:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def score_text(self, text):
        """Returns the 'compound score' between -1 (Negative) and +1 (Positive)"""
        if not text: return 0.0
        scores = self.analyzer.polarity_scores(text)
        return scores['compound']
    
    def analyze_dataframe(self, df, text_column='title'):
        """Applies the analysis to the entire DataFrame"""
        df = df.copy()
        # Apply the model to each title
        df['sentiment_score'] = df[text_column].apply(self.score_text)
        
        # Simple classification for visualization
        df['sentiment_label'] = df['sentiment_score'].apply(
            lambda x: 'Bullish 🟢' if x > 0.05 else ('Bearish 🔴' if x < -0.05 else 'Neutral ⚪')
        )
        return df