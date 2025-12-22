# src/evaluation/compare.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

class ModelComparator:
    """
    The 'Judge' class. Compares two volatility models using:
    1. Statistical Tests (Diebold-Mariano)
    2. Risk Metrics (QLIKE, RMSE)
    3. Regime Visualization (Cumulative Loss)
    """
    def __init__(self, y_true, model_a_pred, model_b_pred, name_a="GARCH", name_b="LSTM"):
        self.y_true = np.array(y_true)
        self.pred_a = np.array(model_a_pred)
        self.pred_b = np.array(model_b_pred)
        self.name_a = name_a
        self.name_b = name_b
        
        # Ensure positivity for QLIKE
        self.pred_a = np.maximum(self.pred_a, 1e-6)
        self.pred_b = np.maximum(self.pred_b, 1e-6)

    def _qlike_loss(self, y_true, y_pred):
        """Calculates vector of QLIKE losses: log(h) + y^2/h"""
        # Input assumed to be Variance. If inputs are Vol, square them.
        # Let's assume inputs are ANNUALIZED VOLATILITY for interpretation,
        # so we convert to Daily Variance for the loss calculation.
        
        # Convert Ann Vol -> Daily Variance: (Vol / sqrt(252))^2
        var_pred = (y_pred / np.sqrt(252)) ** 2
        var_true = (y_true / np.sqrt(252)) ** 2
        
        return np.log(var_pred) + (var_true / var_pred)

    def diebold_mariano_test(self):
        """
        Performs the Diebold-Mariano test to check if Model A is significantly better than B.
        Returns: DM Statistic, p-value
        """
        # 1. Calculate Loss Series for both models (Using QLIKE as the standard)
        loss_a = self._qlike_loss(self.y_true, self.pred_a)
        loss_b = self._qlike_loss(self.y_true, self.pred_b)
        
        # 2. Define Differential: d = Loss(A) - Loss(B)
        # If d < 0, it means Loss(A) was smaller (A is better)
        d = loss_a - loss_b
        
        # 3. Compute Mean and Variance of d
        mean_d = np.mean(d)
        var_d = np.var(d, ddof=1)
        
        # 4. DM Statistic = Mean / Standard_Error
        # Note: Strictly speaking, we should correct for autocorrelation in d (Newey-West),
        # but for a simple prototype, standard error is an acceptable approximation.
        dm_stat = mean_d / np.sqrt(var_d / len(d))
        
        # 5. p-value (Two-sided)
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
        
        return dm_stat, p_value

    def generate_report(self):
        """Prints the verdict."""
        dm_stat, p_val = self.diebold_mariano_test()
        
        print(f"\n📢 FINAL VERDICT: {self.name_a} vs {self.name_b}")
        print("-" * 50)
        print(f"Diebold-Mariano Statistic: {dm_stat:.4f}")
        print(f"P-Value: {p_val:.4f}")
        
        if p_val < 0.05:
            if dm_stat < 0:
                print(f"✅ RESULT: {self.name_a} is STATISTICALLY BETTER than {self.name_b} (95% Conf).")
            else:
                print(f"❌ RESULT: {self.name_b} is STATISTICALLY BETTER than {self.name_a} (95% Conf).")
        else:
            print(f"⚖️ RESULT: No significant difference found. Models are tied.")
            
    def plot_cumulative_loss_diff(self):
        """
        Visualizes WHEN one model outperforms the other.
        Plotting Cumulative(Loss_A - Loss_B).
        - Downward Slope = Model A is winning (Lower Loss).
        - Upward Slope = Model B is winning.
        """
        loss_a = self._qlike_loss(self.y_true, self.pred_a)
        loss_b = self._qlike_loss(self.y_true, self.pred_b)
        
        diff = np.cumsum(loss_a - loss_b)
        
        plt.figure(figsize=(12, 5))
        plt.plot(diff, label=f"Cumulative Loss ({self.name_a} - {self.name_b})", color='purple')
        plt.axhline(0, color='black', linestyle='--')
        plt.title(f"Regime Analysis: When does {self.name_a} win?")
        plt.ylabel("Cumulative Loss Difference")
        plt.xlabel("Time (Days)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Annotation for interpretation
        plt.text(0, diff.min(), "  Slope DOWN = A is better", fontsize=10, color='green')
        plt.text(0, diff.max(), "  Slope UP = B is better", fontsize=10, color='red')
        
        plt.show()