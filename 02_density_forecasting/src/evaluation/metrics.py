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

def test_pit_uniformity(pit_series: Union[pd.Series, np.ndarray], alpha: float = 0.05) -> Tuple[float, float, str]:
    """
    Performs the Kolmogorov-Smirnov test to check if PIT values are U(0,1).

    Args:
        pit_series: Series or array of PIT values in [0, 1].
        alpha:      Significance level. Default 0.05.

    Returns:
        Tuple of (ks_statistic, p_value, result_string) where result_string
        is 'Reject' or 'Not Reject'.
    """
    clean_pit = pit_series.dropna().values if isinstance(pit_series, pd.Series) else pit_series[~np.isnan(pit_series)]

    ks_stat, p_value = kstest(clean_pit, 'uniform')

    print("\n--- STATISTICAL CALIBRATION TEST (Kolmogorov-Smirnov) ---")
    print(f"K-S Statistic: {ks_stat:.5f}")
    print(f"P-Value:       {p_value:.5e} (Threshold: {alpha})")

    if p_value < alpha:
        print("Result:        ❌ REJECT the Null Hypothesis")
        print("Diagnosis:     The PIT is NOT uniform. The model's predicted distribution")
        print("               does not match the realized market data.")
        result = 'Reject'
    else:
        print("Result:        ✅ FAIL TO REJECT the Null Hypothesis")
        print("Diagnosis:     The PIT is statistically uniform. The model is")
        print("               well-calibrated to the market distribution.")
        result = 'Not Reject'

    return ks_stat, p_value, result


def print_evaluation_summary(df_ret: pd.DataFrame, ticker_name: str) -> None:
    """
    Prints the summary of the quantitative forecasting metrics.
    """
    print(f"\n--- {ticker_name} EVALUATION SUMMARY ---")
    print(f"Average CRPS:           {df_ret['CRPS'].mean():.5f} (Lower is better)")
    print(f"Total Log-Likelihood:   {df_ret['Log_Likelihood'].sum():.2f}")
    print(f"Average Log-Likelihood: {df_ret['Log_Likelihood'].mean():.5f}")


# ---------------------------------------------------------------------------
# Financial Summary of a Forecast Distribution
# ---------------------------------------------------------------------------

def summarize_forecast_distribution(
    samples: np.ndarray,
    rf_daily: float = 0.0,
    periods_per_year: int = 252,
    confidence_levels: tuple = (0.95, 0.99),
) -> pd.DataFrame:
    """
    Computes financial risk and performance statistics from a distribution of
    forecasted returns (an ensemble / Monte-Carlo sample array).

    Input
    -----
    samples : np.ndarray, shape (N,)
        Simulated/forecasted daily return samples for a single time step t+1.
    rf_daily : float
        Daily risk-free rate (default 0 — use e.g. 0.05/252 for 5 % p.a.).
    periods_per_year : int
        Trading periods in a year used for annualisation (default 252).
    confidence_levels : tuple of float
        Confidence levels for VaR / CVaR (default (0.95, 0.99)).

    Returns
    -------
    pd.DataFrame
        Single-column DataFrame indexed by metric name, easy to display()
        or join with other model summaries.
    """
    s = np.asarray(samples, dtype=float)
    s = s[~np.isnan(s)]          # drop NaNs defensively
    ann = np.sqrt(periods_per_year)

    rows = {}

    # ── Central Tendency ────────────────────────────────────────────────────
    rows["Expected Return (daily)"] = np.mean(s)
    rows["Median Return  (daily)"] = np.median(s)
    rows["Expected Return (ann.)"]  = np.mean(s) * periods_per_year

    # ── Spread ──────────────────────────────────────────────────────────────
    rows["Volatility (daily)"] = np.std(s, ddof=1)
    rows["Volatility (ann.)"]  = np.std(s, ddof=1) * ann

    # ── Shape ───────────────────────────────────────────────────────────────
    rows["Skewness"] = float(pd.Series(s).skew())
    rows["Kurtosis (excess)"] = float(pd.Series(s).kurt())

    # ── Risk / VaR & CVaR ───────────────────────────────────────────────────
    for cl in confidence_levels:
        pct_label = f"{int(cl * 100)}%"
        var  = np.percentile(s, (1 - cl) * 100)
        cvar = s[s <= var].mean() if np.any(s <= var) else var
        rows[f"VaR  {pct_label} (daily)"] = var
        rows[f"CVaR {pct_label} (daily)"] = cvar
        rows[f"VaR  {pct_label} (ann.)"]  = var  * ann
        rows[f"CVaR {pct_label} (ann.)"]  = cvar * ann

    # ── Performance Ratios ──────────────────────────────────────────────────
    excess   = s - rf_daily
    std_down = np.std(s[s < rf_daily], ddof=1) if np.any(s < rf_daily) else np.nan

    sharpe  = (np.mean(excess) / np.std(s, ddof=1)) * ann if np.std(s) > 0 else np.nan
    sortino = (np.mean(excess) / std_down) * ann           if std_down and std_down > 0 else np.nan

    # Omega Ratio: P(gain) weighted mean gain / P(loss) weighted mean loss
    gains  = s[s > rf_daily] - rf_daily
    losses = rf_daily - s[s <= rf_daily]
    omega  = (gains.mean() * len(gains)) / (losses.mean() * len(losses)) if len(losses) > 0 and losses.mean() > 0 else np.nan

    # Tail Ratio: 95th pct gain / abs(5th pct loss)
    tail_ratio = abs(np.percentile(s, 95)) / abs(np.percentile(s, 5)) if np.percentile(s, 5) != 0 else np.nan

    rows["Sharpe Ratio  (ann.)"]  = sharpe
    rows["Sortino Ratio (ann.)"]  = sortino
    rows["Omega Ratio"]           = omega
    rows["Tail Ratio  (p95/p5)"]  = tail_ratio

    # ── Probability Signals ─────────────────────────────────────────────────
    rows["P(Return > 0)"]  = np.mean(s > 0)
    rows["P(Return > rf)"] = np.mean(s > rf_daily)

    df = (
        pd.DataFrame.from_dict(rows, orient="index", columns=["Value"])
        .round(6)
    )
    return df


def rolling_ks_test(pit_series: pd.Series, window: int = 60) -> pd.Series:
    """
    Performs a Rolling Kolmogorov-Smirnov test to detect localized miscalibration.
    
    Args:
        pit_series: pd.Series containing Probability Integral Transform values.
        window: Integer representing the lookback window (e.g., 60 trading days).
        
    Returns:
        pd.Series of p-values over time. When the p-value drops below 0.05,
        the model has locally failed (miscalibrated).
    """
    # Initialize an empty series to store the p-values
    p_values = pd.Series(index=pit_series.index, dtype=float)
    
    # We start calculating only after we have enough data for the first window
    for i in range(window, len(pit_series)):
        window_data = pit_series.iloc[i - window : i].dropna()
        
        # Ensure we have enough valid data points in the window to run the test
        if len(window_data) >= (window * 0.8): 
            _, p_val = kstest(window_data, 'uniform')
            p_values.iloc[i] = p_val
            
    return p_values


def block_ks_test(pit_series: pd.Series, block_size: int = 60, alpha: float = 0.05) -> pd.DataFrame:
    """
    Performs Kolmogorov-Smirnov tests on discrete, NON-OVERLAPPING blocks of time.
    This eliminates the autocorrelation issue inherent in rolling windows.
    
    Args:
        pit_series: pd.Series containing Probability Integral Transform values.
        block_size: Integer representing the number of days per independent block.
        alpha: Significance level for rejection.
        
    Returns:
        pd.DataFrame summarizing the test results for each independent block.
    """
    # Drop NaNs to ensure clean data chunks
    clean_pit = pit_series.dropna()
    total_days = len(clean_pit)
    
    results = []
    
    # Iterate through the series in discrete jumps (block_size)
    for start_idx in range(0, total_days, block_size):
        end_idx = min(start_idx + block_size, total_days)
        block_data = clean_pit.iloc[start_idx:end_idx]
        
        # Only test if we have a full (or nearly full) block
        if len(block_data) >= (block_size * 0.8):
            ks_stat, p_val = kstest(block_data.values, 'uniform')
            
            results.append({
                'Start_Date': block_data.index[0],
                'End_Date': block_data.index[-1],
                'KS_Stat': ks_stat,
                'P_Value': p_val,
                'Status': '❌ FAILED' if p_val < alpha else '✅ PASSED'
            })
            
    df_results = pd.DataFrame(results)
    
    # Calculate the total failure rate
    failure_rate = (df_results['Status'] == '❌ FAILED').mean() * 100
    print(f"\n--- INDEPENDENT BLOCK TEST SUMMARY ({block_size}-Day Windows) ---")
    print(f"Total Independent Blocks: {len(df_results)}")
    print(f"Blocks Failed (p < {alpha}): {sum(df_results['Status'] == '❌ FAILED')}")
    print(f"True Failure Rate:        {failure_rate:.1f}%")
    
    return df_results
