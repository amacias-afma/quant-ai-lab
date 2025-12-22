import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm

# from statsmodels.stats.diagnostic import acorr_ljungbox

class ParametricVaRModel:
    # model = Model(ticker, model_name, params)

    def __init__(self, model_name=None, market_data=None, window=None, date_limit=None, alpha=0.01):
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
        self.data_train = self.market_data.data.copy()
        self.data_train['mean_ret'] = self.data_train['log_ret'].rolling(window=self.window).mean()
        self.data_train['residual'] = self.data_train['log_ret'] - self.data_train['mean_ret']

        return

    def test(self, df_data_test):
        confidence = norm.ppf(self.alpha)
        self.data_test = df_data_test.copy()
        df_data_full = self.market_data.data.copy()
        df_data_full['mean_ret'] = df_data_full['log_ret'].rolling(window=self.window).mean()
        df_data_full['std_ret'] = df_data_full['log_ret'].rolling(window=self.window).std()
        df_data_full.dropna(inplace=True)
        df_data_full['value_at_risk'] = df_data_full['mean_ret'] + confidence * df_data_full['std_ret']
        df_data_full['value_at_risk'] = df_data_full['value_at_risk'].shift(1)
        self.data_test = self.data_test.join(df_data_full['value_at_risk'])
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