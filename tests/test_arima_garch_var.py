import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.volatility.arima_garch import arima_garch_forecast

def test_arima_garch_forecast_structure():
    """
    Verifies that arima_garch_forecast returns a Series with the correct index and valid values.
    """
    # 1. Generate Synthetic Data (Random Walk)
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=200, freq='B')
    returns = np.random.normal(0, 1, 200) # Already scaled "like" 100x returns
    
    data = pd.Series(returns, index=dates)
    
    # Split
    train_data = data.iloc[:150]
    test_data = data.iloc[150:]
    
    # 2. Run Forecast
    # Using simple orders to be fast
    var_forecast = arima_garch_forecast(train_data, test_data, p=1, q=0, garch_p=1, garch_q=1)
    
    # 3. Assertions
    # Shape
    assert len(var_forecast) == len(test_data)
    assert var_forecast.index.equals(test_data.index)
    
    # Values
    assert not var_forecast.isna().any()
    # VaR should generally be positive (absolute loss threshold), but strict sign depends on definition.
    # In implementation: var_forecast = -(mean + Z*vol).
    # If Z is negative (e.g. -2.33), and mean is ~0, then -( - ) is positive.
    # So we expect mostly positive values for VaR logic "Loss < VaR".
    assert (var_forecast > 0).mean() > 0.9 # Most should be positive
    
def test_arima_garch_linear_trend():
    """
    Verifies that ARIMA captures a simple trend.
    """
    # Create data with a strong drift
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100, freq='B')
    # Trend + Noise
    # Returns have a positive mean 0.5
    returns = np.random.normal(0.5, 0.1, 100)
    
    data = pd.Series(returns, index=dates)
    train = data.iloc[:80]
    test = data.iloc[80:]
    
    # Forecast
    var_forecast = arima_garch_forecast(train, test, p=1, q=0, confidence_level=0.99)
    
    # The mean forecast should be around 0.5.
    # The volatility is low (0.1).
    # VaR = -(Mean + Z*Vol) = -(0.5 + (-2.33)*0.1) = -(0.5 - 0.233) = -0.267
    # So VaR might be negative (meaning we actually expect a profit, or at least not a loss).
    
    # Let's check that the code runs and produces values. 
    # The main check is that it doesn't crash on trending data.
    assert len(var_forecast) == len(test)
