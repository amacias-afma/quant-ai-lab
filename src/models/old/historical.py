import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

class HistoricalVolModel:
    def __init__(self, ticker='SPY', data=None, window=22):
        self.ticker = ticker
        self.data = data.copy()
        self.window = window
        self.name = f"HistVol_{window}d"
        self.z_scores = None # Standardized residuals

    def train(self):
        """
        'Training' for historical vol is just calculating the rolling statistics.
        We shift by 1 because Vol(t) is calculated using data up to t, 
        and is used to predict Return(t+1).
        """
        # Calculate Rolling Volatility (Annualized for reference, but we keep daily for z-scores)
        # Note: We compute 'sigma_t' which is the estimate for 'r_{t+1}'
        self.data[f'vol_{self.window}'] = self.data['log_ret'].rolling(window=self.window).std()
        
        # Shift 1: The vol calculated at end of day T is the forecast for T+1
        self.data[f'sigma_pred'] = self.data[f'vol_{self.window}'].shift(1)
        
        # Calculate Mean (Drift)
        self.data[f'mu_{self.window}'] = self.data['log_ret'].rolling(window=self.window).mean()
        self.data[f'mu_pred'] = self.data[f'mu_{self.window}'].shift(1)
        
        # Drop NaNs created by window
        self.data.dropna(inplace=True)

    def evaluate_residuals(self):
        """
        Checks if the model successfully 'standardized' the data.
        Invariants z_t = (r_t - mu_t) / sigma_t
        """
        # Calculate Invariants
        r_t = self.data['log_ret']
        mu_t = self.data[f'mu_pred']
        sigma_t = self.data[f'sigma_pred']
        
        # Z-score (Standardized Residuals)
        self.z_scores = (r_t - mu_t) / sigma_t
        
        print(f"\n--- Diagnostic Report: {self.name} ---")
        
        # 1. Normality Test (Jarque-Bera)
        # If Prob < 0.05, it is NOT Normal (Fat tails remain)
        jb_stat, jb_p = stats.jarque_bera(self.z_scores)
        kurt = stats.kurtosis(self.z_scores)
        print(f"1. Normality (JB): p={jb_p:.4f} | Kurtosis={kurt:.2f} (Target=0)")
        if jb_p < 0.05:
            print("   [FAIL] Residuals are NOT Normal. Fat tails persist.")
        else:
            print("   [PASS] Residuals look Normal.")

        # 2. Autocorrelation (Ljung-Box on Squared Residuals)
        # Checks if volatility clustering was removed
        lb_df = acorr_ljungbox(self.z_scores**2, lags=[10], return_df=True)
        lb_p = lb_df['lb_pvalue'].values[0]
        print(f"2. Independence (LB on z^2): p={lb_p:.4f}")
        if lb_p < 0.05:
            print("   [FAIL] Volatility clustering still present (Model is too slow).")
        else:
            print("   [PASS] Volatility clustering successfully removed.")
            
        return self.z_scores

    def get_forecast(self):
        """Returns the annualized volatility forecast series."""
        return self.data[f'sigma_pred'] * np.sqrt(252)