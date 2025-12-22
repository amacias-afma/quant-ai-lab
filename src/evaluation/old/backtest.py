# src/evaluation/backtest.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

class VolatilityBacktester:
    """
    Model-Agnostic Backtesting Engine.
    Compares Forecasts (y_pred) vs Realized Volatility (y_true).
    """
    def __init__(self, y_true, y_pred, model_name="Model"):
        """
        :param y_true: Series/Array of Realized Volatility (Target)
        :param y_pred: Series/Array of Forecasted Volatility
        :param model_name: Label for the plots
        """
        # Ensure alignment
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.model_name = model_name
        
        if len(self.y_true) != len(self.y_pred):
            raise ValueError(f"Shape Mismatch: True {len(self.y_true)} vs Pred {len(self.y_pred)}")

    def qlike_loss(self):
        """
        Calculates QLIKE Loss (Quasi-Likelihood).
        Standard metric for volatility proxy comparison.
        Loss = log(h) + y/h (simplified version for ranking)
        """
        # We assume input is Volatility (std), but QLIKE works on Variance (sigma^2)
        # Convert to variance
        realized_var = self.y_true**2
        pred_var = self.y_pred**2
        
        # Avoid division by zero
        eps = 1e-8
        
        # Loss = sum( log(pred_var) + realized_var / pred_var )
        loss = np.log(pred_var + eps) + (realized_var / (pred_var + eps))
        return np.mean(loss)

    def compute_metrics(self):
        """Returns a dictionary of error metrics."""
        rmse = np.sqrt(mean_squared_error(self.y_true, self.y_pred))
        mae = mean_absolute_error(self.y_true, self.y_pred)
        qlike = self.qlike_loss()
        
        metrics = {
            "Model": self.model_name,
            "RMSE": rmse,
            "MAE": mae,
            "QLIKE": qlike
        }
        
        print(f"\n--- Backtest Report: {self.model_name} ---")
        print(f"RMSE  (Lower is better): {rmse:.4f}")
        print(f"MAE   (Lower is better): {mae:.4f}")
        print(f"QLIKE (Risk Metric)    : {qlike:.4f}")
        
        return metrics

    def plot_results(self, title="Out-of-Sample Backtest"):
        """Generates the comparison chart with smoothing for readability."""
        plt.figure(figsize=(12, 6))
        
        # 1. Plot the "Noisy" Truth (Squared Returns) - heavily transparent
        # We convert variance to annualized volatility for the plot: sqrt(var) * sqrt(252)
        true_vol_daily = np.sqrt(self.y_true) * np.sqrt(252)
        plt.plot(true_vol_daily, label='True Daily Vol (Noise)', color='gray', alpha=0.2)
        
        # 2. Plot a "Smoothed" Truth (21-day rolling of the squared returns)
        # This shows the trend without the noise, making it comparable to GARCH visually
        true_series = pd.Series(self.y_true)
        smooth_truth = np.sqrt(true_series.rolling(window=21).mean()) * np.sqrt(252)
        plt.plot(smooth_truth, label='True Vol (21d Smoothed)', color='black', linewidth=1.5, linestyle='--')

        # 3. Plot the Model Forecast
        # GARCH output is already variance? Or Vol? 
        # Note: If y_pred is passed as Variance, convert it. 
        # If passed as Vol, keep it. 
        # Let's assume input is Annualized Volatility for consistency with previous steps.
        plt.plot(self.y_pred, label=f'{self.model_name} Forecast', color='blue', linewidth=2)
        
        plt.title(f"{title} - {self.model_name}")
        plt.ylabel("Annualized Volatility")
        plt.xlabel("Time (Trading Days)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    # def plot_results(self, title="Out-of-Sample Backtest"):
    #     """Generates the comparison chart."""
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(self.y_true, label='Realized Vol (Target)', color='gray', alpha=0.6)
    #     plt.plot(self.y_pred, label=f'{self.model_name} Forecast', color='blue', linewidth=1.5)
        
    #     plt.title(f"{title} - {self.model_name}")
    #     plt.ylabel("Annualized Volatility")
    #     plt.xlabel("Time (Trading Days)")
    #     plt.legend()
    #     plt.grid(True, alpha=0.3)
    #     plt.show()