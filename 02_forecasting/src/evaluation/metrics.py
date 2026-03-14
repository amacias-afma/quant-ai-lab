"""
Quantitative Evaluation Metrics for Density Forecasting.

This module provides non-parametric evaluation functions to assess the 
quality of distributional forecasts (ensembles/samples) against realized 
market observations.

Core Metrics:
- Probability Integral Transform (PIT)
- Continuous Ranked Probability Score (CRPS)
- Log-Likelihood (via Kernel Density Estimation)
- Kolmogorov-Smirnov (K-S) Test for Uniformity
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple
from scipy.stats import kstest, gaussian_kde

def calculate_pit(actual: float, samples: np.ndarray) -> float:
    """
    Calculates the Empirical Probability Integral Transform (PIT).
    Represents the proportion of forecasted samples that fall below the actual realization.
    """
    return np.mean(samples <= actual)

def calculate_crps_ensemble(actual: float, samples: np.ndarray) -> float:
    """
    Calculates the Continuous Ranked Probability Score (CRPS) for an ensemble.
    CRPS = Mean Absolute Error (Accuracy) - Expected Spread (Sharpness)
    
    A lower score indicates a better calibrated and sharper forecast.
    """
    # 1. Accuracy penalty (Distance from actual)
    mae = np.mean(np.abs(samples - actual))
    
    # 2. Sharpness reward (Expected spread of the distribution)
    # Sorting allows for O(N log N) computation of the expected difference between samples
    samples_sorted = np.sort(samples)
    N = len(samples)
    spread = 2 * np.sum((np.arange(1, N + 1) - 0.5) * samples_sorted) / (N ** 2)
    
    return mae - spread

def calculate_log_likelihood_kde(actual: float, samples: np.ndarray) -> float:
    """
    Estimates the Log-Likelihood of the actual observation given the predicted samples
    using Gaussian Kernel Density Estimation (KDE).
    """
    # Edge case: If the model predicts the exact same value for all samples (0 variance)
    if np.var(samples) < 1e-10:
        return np.nan
        
    try:
        kde = gaussian_kde(samples)
        # Add a tiny epsilon to prevent log(0) if the actual is far in the tail
        pdf_value = kde.evaluate(actual)[0] + 1e-10 
        return np.log(pdf_value)
    except np.linalg.LinAlgError:
        # Handles rare KDE convergence failures
        return np.nan

def evaluate_forecasts(actual_returns: Union[pd.Series, np.ndarray], 
                       predicted_ensembles: np.ndarray) -> pd.DataFrame:
    """
    Iterates through the time series to compute evaluation metrics for each step.
    
    Args:
        actual_returns: Array or Series of realized market returns. (Shape: T,)
        predicted_ensembles: 2D array of predicted samples. (Shape: T, N_Samples)
        
    Returns:
        pd.DataFrame containing Realized returns, PIT, CRPS, and Log_Likelihood.
    """
    T = len(actual_returns)
    assert predicted_ensembles.shape[0] == T, "Time dimension mismatch between actuals and predictions."
    
    results = {
        'Realized': np.zeros(T),
        'PIT': np.zeros(T),
        'CRPS': np.zeros(T),
        'Log_Likelihood': np.zeros(T)
    }
    
    for t in range(T):
        actual = actual_returns.iloc[t] if isinstance(actual_returns, pd.Series) else actual_returns[t]
        samples = predicted_ensembles[t]
        
        results['Realized'][t] = actual
        results['PIT'][t] = calculate_pit(actual, samples)
        results['CRPS'][t] = calculate_crps_ensemble(actual, samples)
        results['Log_Likelihood'][t] = calculate_log_likelihood_kde(actual, samples)
        
    # Use the original index if a pandas Series was provided
    idx = actual_returns.index if isinstance(actual_returns, pd.Series) else range(T)
    return pd.DataFrame(results, index=idx)

def test_pit_uniformity(pit_series: Union[pd.Series, np.ndarray], alpha: float = 0.05) -> Tuple[float, float, bool]:
    """
    Performs the Kolmogorov-Smirnov test to check if PIT values are U(0,1).
    
    Returns:
        Tuple containing (KS_Statistic, P_Value, Is_Calibrated_Boolean)
    """
    clean_pit = pit_series.dropna() if isinstance(pit_series, pd.Series) else pit_series[~np.isnan(pit_series)]
    
    ks_stat, p_value = kstest(clean_pit, 'uniform')
    is_calibrated = p_value >= alpha
    
    print("\n--- STATISTICAL CALIBRATION TEST (Kolmogorov-Smirnov) ---")
    print(f"K-S Statistic: {ks_stat:.5f}")
    print(f"P-Value:       {p_value:.5e} (Threshold: {alpha})")
    
    if is_calibrated:
        print("Result:        ✅ PASS (Fail to Reject Null)")
        print("Diagnosis:     The model is Statistically Calibrated.")
    else:
        print("Result:        ❌ REJECT Null")
        print("Diagnosis:     The model is Miscalibrated (PIT is not Uniform).")
        
    return ks_stat, p_value, is_calibrated