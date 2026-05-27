"""
Quantitative Evaluation Metrics for Density Forecasting.

This module provides non-parametric evaluation functions to assess the
quality of distributional forecasts (ensembles / Monte-Carlo samples)
against realized market observations.

Pipeline
--------
1. Single-step metrics (one time step at a time):
   - calculate_pit               — Probability Integral Transform
   - calculate_crps_ensemble     — Continuous Ranked Probability Score
   - calculate_log_likelihood_kde — Log-Likelihood via KDE

2. Batch evaluation (loop over the full time series):
   - evaluate_forecasts          — Applies the three metrics above to every row

3. Calibration tests (statistical):
   - test_pit_uniformity         — Global Kolmogorov-Smirnov test on PIT values
   - rolling_ks_test             — Rolling K-S p-values (overlapping windows)
   - block_ks_test               — Block K-S tests (non-overlapping windows)

4. Display helpers:
   - print_evaluation_summary    — Prints average CRPS and Log-Likelihood

5. Financial summary:
   - summarize_forecast_distribution — Risk/return stats from a sample array

6. Composite utilities:
   - calculate_fail_rate         — End-to-end pipeline: align → evaluate → block KS
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple
from scipy.stats import kstest, gaussian_kde


# ---------------------------------------------------------------------------
# 1. Single-step Metrics
# ---------------------------------------------------------------------------

def calculate_pit(actual: float, samples: np.ndarray) -> float:
    """
    Empirical Probability Integral Transform (PIT).

    Returns the proportion of forecasted samples that fall at or below the
    actual realized value.  If the model is perfectly calibrated, PIT values
    collected over time should follow a Uniform(0, 1) distribution.

    Args:
        actual:  Realized return at time t.
        samples: Forecast ensemble for time t. Shape: (N,)

    Returns:
        float in [0, 1].
    """
    return np.mean(samples <= actual)


def calculate_crps_ensemble(actual: float, samples: np.ndarray) -> float:
    """
    Continuous Ranked Probability Score (CRPS) for an ensemble forecast.

    CRPS = Accuracy penalty − Sharpness reward
         = E[|X - y|] − ½ · E[|X - X'|]

    where X, X' are independent draws from the forecast distribution and y
    is the realized value.  A *lower* CRPS indicates a better-calibrated and
    sharper forecast.

    The sharpness term is computed in O(N log N) via a sorted-index trick,
    avoiding the O(N²) naive double-loop.

    Args:
        actual:  Realized return at time t.
        samples: Forecast ensemble for time t. Shape: (N,)

    Returns:
        float — CRPS score (lower is better).
    """
    # Accuracy: mean absolute distance from actual
    mae = np.mean(np.abs(samples - actual))

    # Sharpness: expected spread of the distribution (O(N log N) formula)
    samples_sorted = np.sort(samples)
    N      = len(samples)
    spread = 2 * np.sum((np.arange(1, N + 1) - 0.5) * samples_sorted) / (N ** 2)

    return mae - spread


def calculate_log_likelihood_kde(actual: float, samples: np.ndarray) -> float:
    """
    Log-Likelihood of the realized observation given the forecast ensemble.

    Estimates the forecast PDF at `actual` using Gaussian Kernel Density
    Estimation (KDE) applied to `samples`, then returns log(PDF + ε).

    A tiny epsilon (1e-10) is added before taking the log to avoid -inf when
    the actual falls deep in the tail of a sparse ensemble.

    Args:
        actual:  Realized return at time t.
        samples: Forecast ensemble for time t. Shape: (N,)

    Returns:
        float — log-likelihood (higher is better). Returns np.nan if the KDE
        cannot be computed (zero-variance ensemble or numerical failure).
    """
    # Degenerate case: all samples identical → KDE bandwidth collapses
    if np.var(samples) < 1e-10:
        return np.nan

    try:
        kde       = gaussian_kde(samples)
        pdf_value = kde.evaluate(actual)[0] + 1e-10
        return np.log(pdf_value)
    except np.linalg.LinAlgError:
        return np.nan


# ---------------------------------------------------------------------------
# 2. Batch Evaluation
# ---------------------------------------------------------------------------

def evaluate_forecasts(
    actual_returns: Union[pd.Series, np.ndarray],
    predicted_ensembles: np.ndarray,
) -> pd.DataFrame:
    """
    Compute PIT, CRPS, and Log-Likelihood for every time step in the series.

    Args:
        actual_returns:       Realized returns. Shape: (T,)
        predicted_ensembles:  Forecast ensembles. Shape: (T, N_samples)

    Returns:
        pd.DataFrame with columns ['Realized', 'PIT', 'CRPS', 'Log_Likelihood'],
        indexed by the original index of `actual_returns` if it is a Series.
    """
    T = len(actual_returns)
    assert predicted_ensembles.shape[0] == T, (
        "Time dimension mismatch: actual_returns and predicted_ensembles must have the same length."
    )

    results = {
        'Realized':       np.zeros(T),
        'PIT':            np.zeros(T),
        'CRPS':           np.zeros(T),
        'Log_Likelihood': np.zeros(T),
    }

    for t in range(T):
        actual  = actual_returns.iloc[t] if isinstance(actual_returns, pd.Series) else actual_returns[t]
        samples = predicted_ensembles[t]

        results['Realized'][t]       = actual
        results['PIT'][t]            = calculate_pit(actual, samples)
        results['CRPS'][t]           = calculate_crps_ensemble(actual, samples)
        results['Log_Likelihood'][t] = calculate_log_likelihood_kde(actual, samples)

    idx = actual_returns.index if isinstance(actual_returns, pd.Series) else range(T)
    return pd.DataFrame(results, index=idx)


# ---------------------------------------------------------------------------
# 3. Calibration Tests
# ---------------------------------------------------------------------------

def test_pit_uniformity(
    pit_series: Union[pd.Series, np.ndarray],
    alpha: float = 0.05,
) -> Tuple[float, float, str]:
    """
    Kolmogorov-Smirnov test for PIT uniformity (global calibration check).

    H₀: PIT values are i.i.d. Uniform(0, 1) — the model is well-calibrated.
    H₁: The distribution of PIT values departs from uniform — the model is
        systematically wrong (biased, too wide, or too narrow).

    Args:
        pit_series: PIT values in [0, 1]. Shape: (T,)
        alpha:      Significance level for the test. Default 0.05.

    Returns:
        Tuple of (ks_statistic, p_value, result) where result is 'Reject'
        or 'Not Reject'.
    """
    clean_pit = (
        pit_series.dropna().values
        if isinstance(pit_series, pd.Series)
        else pit_series[~np.isnan(pit_series)]
    )

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


def rolling_ks_test(pit_series: pd.Series, window: int = 60) -> pd.Series:
    """
    Rolling Kolmogorov-Smirnov test to detect *localized* miscalibration.

    A window that slides one day at a time.  When the p-value drops below
    0.05 in a given window, the model has locally lost calibration in that
    period.

    Note: Overlapping windows introduce autocorrelation.  Use `block_ks_test`
    for an independent-blocks version that avoids this issue.

    Args:
        pit_series: PIT values. Shape: (T,)
        window:     Lookback window in trading days (e.g. 60 ≈ 3 months).

    Returns:
        pd.Series of p-values indexed by `pit_series.index`.  Values before
        the first full window are NaN.
    """
    p_values = pd.Series(index=pit_series.index, dtype=float)

    for i in range(window, len(pit_series)):
        window_data = pit_series.iloc[i - window : i].dropna()
        if len(window_data) >= (window * 0.8):
            _, p_val       = kstest(window_data, 'uniform')
            p_values.iloc[i] = p_val

    return p_values


def block_ks_test(
    pit_series: pd.Series,
    block_size: int = 60,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Non-overlapping Block Kolmogorov-Smirnov test.

    Splits the PIT series into discrete, independent blocks of `block_size`
    days and runs a K-S uniformity test on each block separately.  Because
    the blocks do not overlap, the tests are (approximately) independent —
    eliminating the autocorrelation problem of the rolling K-S test.

    Args:
        pit_series: PIT values. Shape: (T,)
        block_size: Number of trading days per independent block.
        alpha:      Significance level for rejection.

    Returns:
        pd.DataFrame with columns: Start_Date, End_Date, KS_Stat, P_Value, Status.
    """
    clean_pit  = pit_series.dropna()
    total_days = len(clean_pit)
    results    = []

    for start_idx in range(0, total_days, block_size):
        end_idx    = min(start_idx + block_size, total_days)
        block_data = clean_pit.iloc[start_idx:end_idx]

        # Skip partial blocks with fewer than 80 % of the required days
        if len(block_data) >= (block_size * 0.8):
            ks_stat, p_val = kstest(block_data.values, 'uniform')
            results.append({
                'Start_Date': block_data.index[0],
                'End_Date':   block_data.index[-1],
                'KS_Stat':    ks_stat,
                'P_Value':    p_val,
                'Status':     '❌ FAILED' if p_val < alpha else '✅ PASSED',
            })

    df_results   = pd.DataFrame(results)
    failure_rate = (df_results['Status'] == '❌ FAILED').mean() * 100

    print(f"\n--- INDEPENDENT BLOCK TEST SUMMARY ({block_size}-Day Windows) ---")
    print(f"Total Independent Blocks: {len(df_results)}")
    print(f"Blocks Failed (p < {alpha}): {sum(df_results['Status'] == '❌ FAILED')}")
    print(f"True Failure Rate:        {failure_rate:.1f}%")

    return df_results


# ---------------------------------------------------------------------------
# 4. Display Helpers
# ---------------------------------------------------------------------------

def print_evaluation_summary(df_ret: pd.DataFrame, ticker_name: str) -> None:
    """
    Print a concise summary of the quantitative forecasting metrics.

    Args:
        df_ret:      DataFrame returned by `evaluate_forecasts`.
        ticker_name: Label shown in the header (e.g. 'BTC-USD').
    """
    print(f"\n--- {ticker_name} EVALUATION SUMMARY ---")
    print(f"Average CRPS:           {df_ret['CRPS'].mean():.5f} (Lower is better)")
    print(f"Total Log-Likelihood:   {df_ret['Log_Likelihood'].sum():.2f}")
    print(f"Average Log-Likelihood: {df_ret['Log_Likelihood'].mean():.5f}")


# ---------------------------------------------------------------------------
# 5. Financial Summary
# ---------------------------------------------------------------------------

def summarize_forecast_distribution(
    samples: np.ndarray,
    rf_daily: float = 0.0,
    periods_per_year: int = 252,
    confidence_levels: tuple = (0.95, 0.99),
) -> pd.DataFrame:
    """
    Financial risk and performance statistics from a forecast sample array.

    Computes a comprehensive table of risk/return metrics from a 1-D array
    of simulated daily returns for a single time step t+1.  Useful for
    presenting the full distributional picture beyond just the mean.

    Args:
        samples:           Simulated/forecasted daily return samples. Shape: (N,)
        rf_daily:          Daily risk-free rate (default 0). Use e.g. 0.05/252
                           for a 5 % annual rate.
        periods_per_year:  Trading periods per year for annualisation (default 252).
        confidence_levels: Confidence levels for VaR / CVaR (default (0.95, 0.99)).

    Returns:
        pd.DataFrame — single-column DataFrame indexed by metric name.
        Call `.display()` or `print()` directly in a Jupyter cell.
    """
    s   = np.asarray(samples, dtype=float)
    s   = s[~np.isnan(s)]        # drop NaNs defensively
    ann = np.sqrt(periods_per_year)

    rows = {}

    # Central Tendency
    rows["Expected Return (daily)"] = np.mean(s)
    rows["Median Return  (daily)"]  = np.median(s)
    rows["Expected Return (ann.)"]  = np.mean(s) * periods_per_year

    # Spread
    rows["Volatility (daily)"] = np.std(s, ddof=1)
    rows["Volatility (ann.)"]  = np.std(s, ddof=1) * ann

    # Shape
    rows["Skewness"]         = float(pd.Series(s).skew())
    rows["Kurtosis (excess)"] = float(pd.Series(s).kurt())

    # VaR & CVaR at each confidence level
    for cl in confidence_levels:
        pct_label = f"{int(cl * 100)}%"
        var  = np.percentile(s, (1 - cl) * 100)
        cvar = s[s <= var].mean() if np.any(s <= var) else var
        rows[f"VaR  {pct_label} (daily)"] = var
        rows[f"CVaR {pct_label} (daily)"] = cvar
        rows[f"VaR  {pct_label} (ann.)"]  = var  * ann
        rows[f"CVaR {pct_label} (ann.)"]  = cvar * ann

    # Performance Ratios
    excess   = s - rf_daily
    std_down = np.std(s[s < rf_daily], ddof=1) if np.any(s < rf_daily) else np.nan

    sharpe  = (np.mean(excess) / np.std(s, ddof=1)) * ann if np.std(s) > 0 else np.nan
    sortino = (np.mean(excess) / std_down) * ann           if std_down and std_down > 0 else np.nan

    # Omega Ratio: weighted mean gain / weighted mean loss relative to rf
    gains  = s[s > rf_daily] - rf_daily
    losses = rf_daily - s[s <= rf_daily]
    omega  = (
        (gains.mean() * len(gains)) / (losses.mean() * len(losses))
        if len(losses) > 0 and losses.mean() > 0
        else np.nan
    )

    # Tail Ratio: 95th-percentile gain / abs(5th-percentile loss)
    p5, p95     = np.percentile(s, 5), np.percentile(s, 95)
    tail_ratio  = abs(p95) / abs(p5) if p5 != 0 else np.nan

    rows["Sharpe Ratio  (ann.)"] = sharpe
    rows["Sortino Ratio (ann.)"] = sortino
    rows["Omega Ratio"]          = omega
    rows["Tail Ratio  (p95/p5)"] = tail_ratio

    # Probability Signals
    rows["P(Return > 0)"]  = np.mean(s > 0)
    rows["P(Return > rf)"] = np.mean(s > rf_daily)

    return pd.DataFrame.from_dict(rows, orient="index", columns=["Value"]).round(6)


# ---------------------------------------------------------------------------
# 6. Composite Utilities
# ---------------------------------------------------------------------------

def _align_predictions(
    predictions: np.ndarray,
    df_ret: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align a raw predictions array with the returns DataFrame by matching dates.

    Wraps the predictions array in a DataFrame with the same index as
    `df_ret`, drops rows where predictions are NaN, and filters `df_ret`
    to the matching dates.

    Args:
        predictions: 2D array of shape (T, N_samples).
        df_ret:      Returns DataFrame with a DatetimeIndex and a 'returns' column.

    Returns:
        Tuple of (df_predictions, df_ret_aligned) — both sharing the same index.
    """
    df_predictions = pd.DataFrame(predictions)
    df_predictions.index = df_ret.index
    df_predictions.dropna(inplace=True)
    df_ret_aligned = df_ret[df_ret.index.isin(df_predictions.index)].copy()
    return df_predictions, df_ret_aligned


def calculate_fail_rate(
    predictions: np.ndarray,
    df_ret: pd.DataFrame,
    block_size: int,
    window: int,
    model: str,
    alpha: float = 0.05,
) -> float:
    """
    End-to-end failure rate: align predictions → evaluate → run block K-S test.

    Args:
        predictions: Raw predictions array. Shape: (T, N_samples)
        df_ret:      Returns DataFrame with a 'returns' column.
        block_size:  Block size passed to `block_ks_test`.
        window:      Warm-up window (skipped for GARCH models, which need extra
                     rows before producing valid forecasts).
        model:       Model identifier string. Pass 'Garch' to skip the first
                     `window` rows of the evaluation.
        alpha:       Significance level for the block K-S test.

    Returns:
        float — fraction of blocks that failed the K-S test (0.0 = all pass).
    """
    df_predictions, df_ret_aux = _align_predictions(predictions, df_ret)

    if model == 'Garch':
        df_evaluation = evaluate_forecasts(
            df_ret_aux.iloc[window:]['returns'],
            df_predictions.iloc[window:].values,
        )
    else:
        df_evaluation = evaluate_forecasts(
            df_ret_aux['returns'],
            df_predictions.values,
        )

    df_results         = block_ks_test(df_evaluation['PIT'], block_size=block_size)
    df_results['Fail'] = (df_results['P_Value'] < alpha).astype(int)
    return df_results['Fail'].sum() / len(df_results)
