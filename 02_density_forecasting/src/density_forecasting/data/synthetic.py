import numpy as np
import pandas as pd
from datetime import date, timedelta
from scipy import stats
from typing import Tuple, Optional

def create_synthetic(
    type: str = 'student_t_garch',
    n_days: int = 1500,
    start_date: Optional[date] = None,
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Unified entry point for generating synthetic financial time series.
    
    Args:
        type: The type of synthetic data to generate. 
              Options: 'normal_garch', 'student_t_garch', 'time_varying_normal', 'vix_dependent'
        n_days: Number of days to simulate.
        start_date: Start date for the index.
        **kwargs: Additional parameters passed to the specific generator.
                  
    Returns:
        Tuple of DataFrames: (df_asset, df_vix)
    """
    if type == 'normal_garch':
        return _generate_garch(n_days, start_date, innovation_dist='normal', **kwargs)
    elif type == 'student_t_garch':
        return _generate_garch(n_days, start_date, innovation_dist='student_t', **kwargs)
    elif type == 'time_varying_normal':
        return _generate_time_varying_normal(n_days, start_date, **kwargs)
    elif type == 'vix_dependent':
        return _generate_vix_dependent(**kwargs)
    else:
        raise ValueError(f"Unknown synthetic type: {type}")

def _generate_vix_dependent(
    vix_series: pd.Series,
    dynamics: str = 'linear',
    mu: float = 0.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate returns where the true volatility (sigma) is a specific function 
    of an existing VIX series.
    
    Args:
        vix_series: The input VIX series (Pandas Series)
        dynamics: 'linear', 'quadratic', or 'sinusoidal'
        mu: Constant mean return
    """
    n_days = len(vix_series)
    dates = vix_series.index
    vix_vals = vix_series.values
    
    if dynamics == 'linear':
        # Linear dependence: sigma scales directly with VIX
        # Normal VIX is ~15-20. We want daily sigma around 0.01.
        sigma_t = 0.002 + 0.0005 * vix_vals
    elif dynamics == 'quadratic':
        # Quadratic dependence
        sigma_t = 0.005 + 0.00003 * (vix_vals ** 2)
    elif dynamics == 'sinusoidal':
        # Sinusoidal dependence based on VIX levels
        # VIX usually ranges from 10 to 40. We scale it so it oscillates.
        sigma_t = 0.015 + 0.010 * np.sin(vix_vals / 5.0)
    else:
        raise ValueError(f"Unknown dynamics: {dynamics}")
        
    # Ensure strict positivity for volatility
    sigma_t = np.maximum(sigma_t, 1e-4)
    
    # Generate Normal returns
    z = np.random.normal(0, 1, size=n_days)
    z = z.reshape(-1, 1)
    sigma_t = sigma_t.reshape(-1, 1)
    returns = mu + sigma_t * z
    
    prices = 100 * np.exp(np.cumsum(returns))
    
    print("sigma_t:", sigma_t.shape)
    print("z:", z.shape)
    print("returns:", returns.shape)
    print("prices:", prices.shape)


    df_asset = pd.DataFrame({
        'prices': prices,
        'returns': returns[:, 0]
    }, index=dates)
    df_asset.index.name = 'Date'
    
    # Return the exact VIX that was passed in
    df_vix = pd.DataFrame({
        'VIX': vix_vals[:, 0]
    }, index=dates)
    df_vix.index.name = 'Date'
    
    return df_asset, df_vix

def _generate_time_varying_normal(
    n_days: int = 1500,
    start_date: Optional[date] = None,
    dynamics: str = 'sinusoidal',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate a Normal distribution where mu and sigma change deterministically 
    over time to test if ML models can capture frequency or linear shifts.
    
    Args:
        dynamics: 'sinusoidal' (periodic shifts) or 'linear' (trend shifts)
    """
    if start_date is None:
        start_date = date.today() - timedelta(days=n_days * 2)
        
    dates = pd.date_range(start=start_date, periods=n_days, freq='B')
    if len(dates) > n_days: dates = dates[:n_days]
    elif len(dates) < n_days: dates = pd.date_range(start=start_date, periods=n_days)
        
    t = np.arange(n_days)
    
    if dynamics == 'sinusoidal':
        # Mu oscillating between -0.002 and 0.002 on a ~6 month (120 day) cycle
        mu_t = 0.002 * np.sin(2 * np.pi * t / 120)
        # Volatility oscillating smoothly between low (0.005) and high (0.02)
        sigma_t = 0.0125 + 0.0075 * np.cos(2 * np.pi * t / 250)
    elif dynamics == 'linear':
        # Mu trending linearly from -0.002 to 0.002 over the whole period
        mu_t = np.linspace(-0.002, 0.002, n_days)
        # Volatility trending linearly upwards
        sigma_t = np.linspace(0.005, 0.025, n_days)
    else:
        raise ValueError(f"Unknown dynamics: {dynamics}")
        
    # Generate returns
    z = np.random.normal(0, 1, size=n_days)
    returns = mu_t + sigma_t * z
    
    prices = 100 * np.exp(np.cumsum(returns))
    
    df_asset = pd.DataFrame({
        'prices': prices,
        'returns': returns
    }, index=dates)
    df_asset.index.name = 'Date'
    
    # VIX is annualized volatility (in percentage points) + noise
    vix = sigma_t * np.sqrt(252) * 100
    vix_noisy = vix + np.random.normal(0, 1.0, size=n_days)
    vix_noisy = np.maximum(vix_noisy, 5.0)
    
    df_vix = pd.DataFrame({
        'VIX': vix_noisy
    }, index=dates)
    df_vix.index.name = 'Date'
    
    return df_asset, df_vix

def _generate_garch(
    n_days: int = 1500,
    start_date: Optional[date] = None,
    innovation_dist: str = 'student_t',
    mu: float = 0.0,
    ar1: float = 0.05,
    omega: float = 1e-5,
    alpha: float = 0.1,
    beta: float = 0.85,
    nu: float = 5.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate an AR(1)-GARCH(1,1) process with specified innovations.
    """
    if start_date is None:
        start_date = date.today() - timedelta(days=n_days * 2) # *2 to account for business days approx
    
    # Generate dates (approximate trading days)
    dates = pd.date_range(start=start_date, periods=n_days, freq='B')
    if len(dates) > n_days:
        dates = dates[:n_days]
    elif len(dates) < n_days:
        # Just in case pd.date_range gives fewer periods
        dates = pd.date_range(start=start_date, periods=n_days)
    
    # Initialize arrays
    returns = np.zeros(n_days)
    sigma2 = np.zeros(n_days)
    
    # Generate innovations and determine unconditional variance for initialization
    if innovation_dist == 'student_t':
        # If nu > 2, Var(t_nu) = nu / (nu - 2). 
        z_var = nu / (nu - 2) if nu > 2 else 1.0
        scale = np.sqrt((nu - 2) / nu) if nu > 2 else 1.0
        z = stats.t.rvs(df=nu, size=n_days) * scale
    elif innovation_dist == 'normal':
        z_var = 1.0
        z = np.random.normal(0, 1, size=n_days)
    else:
        raise ValueError("innovation_dist must be 'student_t' or 'normal'")
    
    # GARCH unconditional variance
    if (alpha * z_var + beta) < 1:
        initial_sigma2 = omega / (1 - alpha * z_var - beta) 
    else:
        initial_sigma2 = omega / (1 - beta)
    
    sigma2[0] = initial_sigma2
    returns[0] = mu + np.sqrt(sigma2[0]) * z[0]
    
    for t in range(1, n_days):
        sigma2[t] = omega + alpha * ((returns[t-1] - mu - ar1 * (returns[t-2] if t>1 else 0))**2) + beta * sigma2[t-1]
        returns[t] = mu + ar1 * returns[t-1] + np.sqrt(sigma2[t]) * z[t]
        
    # Convert to prices
    prices = 100 * np.exp(np.cumsum(returns))
    
    df_asset = pd.DataFrame({
        'prices': prices,
        'returns': returns
    }, index=dates)
    df_asset.index.name = 'Date'
    
    # Create VIX proxy: Annualized conditional volatility in percentage points
    # True volatility is sqrt(sigma2) daily. Annualized is sqrt(252) * sqrt(sigma2)
    # VIX is quoted in percentage points, so * 100.
    vix = np.sqrt(sigma2 * 252) * 100
    
    # Add some noise to VIX so it's not a perfectly deterministic function of the returns
    vix_noisy = vix + np.random.normal(0, 1.0, size=n_days)
    vix_noisy = np.maximum(vix_noisy, 5.0) # VIX is strictly positive, usually > 5
    
    df_vix = pd.DataFrame({
        'VIX': vix_noisy
    }, index=dates)
    df_vix.index.name = 'Date'
    
    return df_asset, df_vix

# Keep the old function signature around for backwards compatibility with notebooks we already updated
def generate_student_t_garch(**kwargs):
    return create_synthetic(type='student_t_garch', **kwargs)
