import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

class HistoricalVaRModel:
    # model = Model(ticker, model_name, params)

    def __init__(self, ticker='SPY', model_name=None, market_data=None, window=None, date_limit=None, alpha=0.01):
        self.ticker = ticker
        self.model_name = model_name
        self.market_data = market_data
        self.date_limit = date_limit
        self.data_test = None
        self.window = window
        self.alpha = alpha

    def train(self):
        """
        'Training' for historical vol is just calculating the rolling statistics.
        We shift by 1 because Vol(t) is calculated using data up to t, 
        and is used to predict Return(t+1).
        """
        # Calculate Rolling Volatility (Annualized for reference, but we keep daily for z-scores)
        # Note: We compute 'sigma_t' which is the estimate for 'r_{t+1}'
        self.market_data.data[f'vol_{self.window}'] = self.market_data.data['log_ret'].rolling(window=self.window).std()
        
        # Shift 1: The vol calculated at end of day T is the forecast for T+1
        self.market_data.data[f'sigma_pred'] = self.market_data.data[f'vol_{self.window}'].shift(1)
        
        # Calculate Mean (Drift)
        self.market_data.data[f'mu_{self.window}'] = self.market_data.data['log_ret'].rolling(window=self.window).mean()
        self.market_data.data[f'mu_pred'] = self.market_data.data[f'mu_{self.window}'].shift(1)
        
        # Drop NaNs created by window
        self.market_data.data.dropna(inplace=True)
        return

    def test(self, df_data_test):
        self.data_test = df_data_test.copy()
        df_data_full = self.market_data.data
        for date in df_data_test.index:
            df_full_aux = df_data_full[(df_data_full.index < date)].tail(self.window)
            value_at_risk = float(df_full_aux.log_ret.quantile(self.alpha))
            self.data_test.loc[date, 'value_at_risk'] = value_at_risk
        self.data_test['outlier'] = 0
        self.data_test.loc[self.data_test.log_ret < self.data_test['value_at_risk'], 'outlier'] = 1
            
    def evaluate_residuals(self):
        """
        Checks if the model successfully 'standardized' the data.
        Invariants z_t = (r_t - mu_t) / sigma_t
        """
        return

    def get_forecast(self):
        """Returns the annualized volatility forecast series."""
        return self.data[f'sigma_pred'] * np.sqrt(252)