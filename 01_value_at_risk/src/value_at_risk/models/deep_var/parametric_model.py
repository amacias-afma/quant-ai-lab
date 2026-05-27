import pandas as pd
import numpy as np
from scipy import stats

def calculate_parametric_var(returns, window=132, alpha=0.05):
    """
    Calculate parametric VaR assuming normal distribution.
    VaR = μ - z_α × σ
    
    Parameters:
    -----------
    returns : pd.Series
        Historical returns
    window : int
        Rolling window for calculating mean and std
    alpha : float
        Significance level (e.g., 0.05 for 95% confidence)
    
    Returns:
    --------
    pd.Series : Parametric VaR estimates
    """
    # Calculate z-score for the given alpha
    z_alpha = abs(stats.norm.ppf(alpha))
    
    # Calculate rolling mean and std
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()
    
    # Parametric VaR = μ - z_α × σ
    parametric_var = rolling_mean - z_alpha * rolling_std
    
    return parametric_var

def priori_value_at_risk(df, rolling=132, alpha=0.05):
    df_value_at_risk = df.copy()
    z0 = float(np.abs(stats.norm.ppf(alpha)))
    
    df_value_at_risk[f'std_{rolling}d'] = df_value_at_risk['log_ret'].rolling(rolling).std()
    df_value_at_risk[f'mean_{rolling}d'] = df_value_at_risk['log_ret'].rolling(rolling).mean()
    df_value_at_risk[f'value_at_risk_hist'] = df_value_at_risk['log_ret'].rolling(rolling).quantile(alpha)
    df_value_at_risk[f'value_at_risk_param'] = df_value_at_risk[f'mean_{rolling}d'] - z0 * df_value_at_risk[f'std_{rolling}d']
    
    return df_value_at_risk