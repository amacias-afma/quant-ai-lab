"""
Baseline Models for Density Forecasting.

These models serve as the "Floor" performance metrics. 
All models adhere to the Non-Parametric standard: they output an ensemble 
(matrix of samples) for each time step, representing the predicted distribution 
for t+1 using only information available up to time t.
"""

import numpy as np
import pandas as pd
from typing import Union
from scipy.stats import norm, t

def historical_simulation(returns: Union[pd.Series, np.ndarray], 
                          window: int = 252, 
                          n_samples: int = 2000) -> np.ndarray:
    """
    Historical Simulation (Non-Parametric Baseline).
    Assumes the future distribution will look exactly like the recent past.
    Samples with replacement from the rolling lookback window.
    
    Args:
        returns: Historical realized returns (Shape: T,)
        window: Number of past days to sample from (e.g., 252 for 1 trading year).
        n_samples: Number of paths/samples to generate for tomorrow's forecast.
        
    Returns:
        np.ndarray of shape (T, n_samples) containing the forecasted distributions.
    """
    ret_array = returns.values if isinstance(returns, pd.Series) else returns
    T = len(ret_array)
    predictions = np.full((T, n_samples), np.nan)
    
    for idx in range(window, T):
        # Strictly use past data to prevent look-ahead bias
        past_window = ret_array[idx - window : idx]
        
        # Sample with replacement
        predictions[idx] = np.random.choice(past_window, size=n_samples, replace=True)
        
    return predictions


def rolling_gaussian(returns: Union[pd.Series, np.ndarray], 
                     window: int = 252, 
                     n_samples: int = 2000) -> np.ndarray:
    """
    Rolling Gaussian / Parametric Normal Baseline.
    Estimates the sample mean and sample standard deviation over a rolling window,
    then draws `n_samples` from a Normal distribution with those parameters.
    
    Args:
        returns: Historical realized returns (Shape: T,)
        window: Lookback period for calculating mu and sigma.
        n_samples: Number of paths to draw.
        
    Returns:
        np.ndarray of shape (T, n_samples)
    """
    ret_array = returns.values if isinstance(returns, pd.Series) else returns
    T = len(ret_array)
    predictions = np.full((T, n_samples), np.nan)
    
    for idx in range(window, T):
        past_window = ret_array[idx - window : idx]
        
        mu = np.mean(past_window)
        sigma = np.std(past_window, ddof=1) # ddof=1 for sample standard deviation
        
        # Draw samples from the estimated Normal distribution
        predictions[idx] = norm.rvs(loc=mu, scale=sigma, size=n_samples)
        
    return predictions


def rolling_student_t(returns: Union[pd.Series, np.ndarray], 
                      window: int = 252, 
                      n_samples: int = 2000) -> np.ndarray:
    """
    Rolling Student-T Baseline.
    The industry-standard upgrade from Gaussian to account for "Fat Tails".
    Fits the degrees of freedom (nu), loc, and scale to the rolling window.
    
    Args:
        returns: Historical realized returns (Shape: T,)
        window: Lookback period for fitting the Student-t parameters.
        n_samples: Number of paths to draw.
        
    Returns:
        np.ndarray of shape (T, n_samples)
    """
    ret_array = returns.values if isinstance(returns, pd.Series) else returns
    T = len(ret_array)
    predictions = np.full((T, n_samples), np.nan)
    
    for idx in range(window, T):
        past_window = ret_array[idx - window : idx]
        
        # Fit the Student-t distribution to the recent past
        # Returns: degrees of freedom (df), mean (loc), standard dev (scale)
        df, loc, scale = t.fit(past_window)
        
        # Draw samples from the estimated Student-t distribution
        predictions[idx] = t.rvs(df=df, loc=loc, scale=scale, size=n_samples)
        
    return predictions