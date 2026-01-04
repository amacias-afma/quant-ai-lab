import pytest
import sys
from unittest.mock import MagicMock

# 1. Mock Google Cloud BigQuery before importing the app
# This prevents the app from trying to connect to GCP during tests
mock_bigquery = MagicMock()
sys.modules["google.cloud"] = MagicMock()
sys.modules["google.cloud.bigquery"] = mock_bigquery

# 2. Import the Flask app
@pytest.fixture
def client():
    """
    Pytest fixture to create a test client for the Flask application.
    This allows us to simulate HTTP requests without running the server.
    """
    # Moved import here to avoid side effects (e.g. NLTK, DB connect) during collection
    from src.app import app
    
    app.config['TESTING'] = True
    app.config['DEBUG'] = False
    
    with app.test_client() as client:
        yield client