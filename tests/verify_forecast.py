
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.lstm import DeepVolEngine

def test_forecast():
    # 1. Setup
    print("Initializing DeepVolEngine...")
    engine = DeepVolEngine(ticker='TEST', seq_len=60)
    
    # 2. Synthetic Data
    # Create 500 days of random returns
    print("Generating synthetic data...")
    dates = pd.date_range(start='2020-01-01', periods=500)
    # Random normal returns approx 1% vol
    returns = np.random.normal(0, 0.01, 500)
    
    df = pd.DataFrame({'price': 100 * np.exp(np.cumsum(returns))}, index=dates)
    df['log_ret'] = returns
    df['target_variance'] = df['log_ret'] ** 2
    
    engine.data = df
    
    # 3. Quick Train (1 epoch)
    print("Quick training (1 epoch)...")
    engine.train(epochs=1)
    
    # 4. Forecast
    n_days = 30
    print(f"Forecasting {n_days} days...")
    mean_vol, p5, p95 = engine.forecast_future(n_days=n_days, n_simulations=50)
    
    print("Forecast shape:", mean_vol.shape)
    print("First 5 values (Mean Vol):", mean_vol[:5])
    
    # Check if values are reasonable (not NaN, positive)
    if np.isnan(mean_vol).any():
        print("ERROR: Forecast contains NaNs!")
    else:
        print("SUCCESS: Forecast values are valid numericals.")

    # 5. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(mean_vol, label='Mean Forecast')
    plt.fill_between(range(n_days), p5, p95, alpha=0.3, label='90% CI')
    plt.title("LSTM N-Day Volatility Forecast (Test)")
    plt.legend()
    plt.savefig("tests/forecast_test.png")
    print("Plot saved to tests/forecast_test.png")

if __name__ == "__main__":
    test_forecast()
