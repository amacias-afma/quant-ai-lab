"""
Baseline Models for Density Forecasting.

These models serve as the "Floor" performance metrics.
All models adhere to the Non-Parametric standard: they output an ensemble
(matrix of samples) for each time step, representing the predicted distribution
for t+1 using only information available up to time t.

Available Models (ordered from simplest to most complex):
    - historical_simulation        : Non-parametric bootstrap from rolling window
    - rolling_gaussian             : Parametric Normal (mean + std from rolling window)
    - rolling_student_t            : Parametric Fat-Tail (Student-t fitted to rolling window)
    - rolling_garch                : Volatility-clustering model (GARCH(1,1) on rolling window)
    - rolling_vix_scaled_gaussian  : Gaussian with VIX-elasticity volatility scaling
    - rolling_vix_scaled_student_t : Student-t with VIX-elasticity volatility scaling
"""

import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.optimize import minimize
from scipy.stats import norm, t
from typing import Union

# Suppress arch convergence warnings for clean notebook output
warnings.filterwarnings('ignore', category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Simple Baselines
# ---------------------------------------------------------------------------

def historical_simulation(
    returns: Union[pd.Series, np.ndarray],
    window: int = 252,
    n_samples: int = 2000,
) -> np.ndarray:
    """
    Historical Simulation — Non-Parametric Baseline.

    Assumes the future distribution will look exactly like the recent past.
    Samples *with replacement* from the rolling lookback window, so the
    output preserves the full empirical shape (fat tails, skewness, etc.)
    without assuming any parametric family.

    Args:
        returns:   Historical realized returns. Shape: (T,)
        window:    Number of past days to sample from (e.g. 252 = 1 trading year).
        n_samples: Number of paths/samples to generate for tomorrow's forecast.

    Returns:
        np.ndarray of shape (T, n_samples). Rows 0…window-1 are NaN (warm-up).
    """
    ret_array = returns.values if isinstance(returns, pd.Series) else returns
    T = len(ret_array)
    predictions = np.full((T, n_samples), np.nan)

    for idx in range(window, T):
        # Strictly past data — no look-ahead bias
        past_window = ret_array[idx - window : idx]
        predictions[idx] = np.random.choice(past_window, size=n_samples, replace=True)

    return predictions


def rolling_gaussian(
    returns: Union[pd.Series, np.ndarray],
    window: int = 252,
    n_samples: int = 2000,
) -> np.ndarray:
    """
    Rolling Gaussian — Parametric Normal Baseline.

    Estimates the sample mean (μ) and sample standard deviation (σ) over a
    rolling window, then draws `n_samples` from N(μ, σ²).  This is the
    textbook parametric benchmark: fast, interpretable, and known to
    underestimate tail risk for financial returns.

    Args:
        returns:   Historical realized returns. Shape: (T,)
        window:    Lookback period for calculating μ and σ.
        n_samples: Number of paths to draw.

    Returns:
        np.ndarray of shape (T, n_samples). Rows 0…window-1 are NaN (warm-up).
    """
    ret_array = returns.values if isinstance(returns, pd.Series) else returns
    T = len(ret_array)
    predictions = np.full((T, n_samples), np.nan)

    for idx in range(window, T):
        past_window = ret_array[idx - window : idx]
        mu    = np.mean(past_window)
        sigma = np.std(past_window, ddof=1)   # sample std
        predictions[idx] = norm.rvs(loc=mu, scale=sigma, size=n_samples)

    return predictions


def rolling_student_t(
    returns: Union[pd.Series, np.ndarray],
    window: int = 252,
    n_samples: int = 2000,
) -> np.ndarray:
    """
    Rolling Student-T — Fat-Tail Parametric Baseline.

    The industry-standard upgrade from Gaussian to account for "fat tails".
    Fits the degrees of freedom (ν), location (μ), and scale (σ) of a
    Student-t distribution to the rolling window via MLE, then draws
    `n_samples` from that distribution.

    A higher ν ≈ Normal; lower ν (e.g. 3–5) implies heavy tails typical
    of financial returns.

    Args:
        returns:   Historical realized returns. Shape: (T,)
        window:    Lookback period for fitting the Student-t parameters.
        n_samples: Number of paths to draw.

    Returns:
        np.ndarray of shape (T, n_samples). Rows 0…window-1 are NaN (warm-up).
    """
    ret_array = returns.values if isinstance(returns, pd.Series) else returns
    T = len(ret_array)
    predictions = np.full((T, n_samples), np.nan)

    for idx in range(window, T):
        past_window = ret_array[idx - window : idx]
        # scipy.stats.t.fit returns (df, loc, scale) via MLE
        df, loc, scale = t.fit(past_window)
        predictions[idx] = t.rvs(df=df, loc=loc, scale=scale, size=n_samples)

    return predictions


# ---------------------------------------------------------------------------
# Volatility-Clustering Baseline
# ---------------------------------------------------------------------------

def rolling_garch(
    returns: pd.Series,
    window: int = 252,
    n_samples: int = 2000,
) -> np.ndarray:
    """
    Rolling GARCH(1,1) — Volatility-Clustering Baseline.

    Fits a GARCH(1,1) model with a Constant Mean and Normal innovations on a
    rolling window to forecast tomorrow's conditional variance, then draws
    `n_samples` from N(μ_t+1, σ²_t+1).

    GARCH captures the key stylized fact that volatility clusters in time
    (calm periods follow calm, turbulent follow turbulent), which the simple
    rolling-window models above miss entirely.

    Args:
        returns:   Historical realized returns. Scaling by 100 (percentages)
                   is recommended for GARCH numerical stability.
        window:    Lookback period for fitting the GARCH parameters.
        n_samples: Number of paths to draw for the forecasted distribution.

    Returns:
        np.ndarray of shape (T, n_samples). Rows 0…window-1 are NaN (warm-up).
    """
    T = len(returns)
    predictions = np.full((T, n_samples), np.nan)

    for idx in range(window, T):
        past_window = returns.iloc[idx - window : idx]
        am = arch_model(past_window, vol='Garch', p=1, q=1,
                        mean='Constant', dist='Normal', rescale=False)
        try:
            res = am.fit(disp='off', show_warning=False)
            forecasts  = res.forecast(horizon=1, reindex=False)
            mu_pred    = forecasts.mean.iloc[-1, 0]
            sigma_pred = np.sqrt(forecasts.variance.iloc[-1, 0])
            predictions[idx] = np.random.normal(loc=mu_pred, scale=sigma_pred, size=n_samples)
        except Exception:
            # Fallback to simple rolling std if GARCH fails to converge
            mu_pred    = np.mean(past_window)
            sigma_pred = np.std(past_window, ddof=1)
            predictions[idx] = np.random.normal(loc=mu_pred, scale=sigma_pred, size=n_samples)

        if idx % 500 == 0:
            print(f"GARCH processing: Day {idx}/{T}")

    return predictions


# ---------------------------------------------------------------------------
# VIX-Adjusted Baselines
# ---------------------------------------------------------------------------

def rolling_vix_scaled_gaussian(
    returns: Union[pd.Series, np.ndarray],
    vix: Union[pd.Series, np.ndarray],
    window: int = 252,
    n_samples: int = 2000,
) -> np.ndarray:
    """
    VIX-Scaled Gaussian — Exogenous Volatility Baseline.

    Extends the Rolling Gaussian by learning an elasticity parameter (β)
    that links historical volatility to the VIX index via Maximum Likelihood
    Estimation (MLE):

        σ_adj(t) = σ_hist · (VIX(t-1) / mean(VIX))^β

    β > 0 means volatility expands when VIX rises (the typical regime).
    β is re-estimated daily on the rolling window using Nelder-Mead with a
    warm-start from the previous day's β for speed.

    Args:
        returns:   Historical realized returns. Shape: (T,)
        vix:       Historical VIX closing prices, aligned with `returns`. Shape: (T,)
        window:    Lookback period for MLE and historical baselines.
        n_samples: Number of paths to draw.

    Returns:
        np.ndarray of shape (T, n_samples). Rows 0…window are NaN (warm-up).
    """
    ret_array = np.asarray(returns, dtype=float).flatten()
    vix_array = np.asarray(vix,     dtype=float).flatten()

    T           = len(ret_array)
    predictions = np.full((T, n_samples), np.nan)
    last_optimal_beta = 0.3
    cold_start        = True

    print(f"Starting VIX-Gaussian optimization for {T - window} days...")

    for idx in range(window + 1, T):
        past_ret       = ret_array[idx - window : idx]
        past_vix_lagged = vix_array[idx - window - 1 : idx - 1]

        mu_train    = float(np.mean(past_ret))
        sigma_train = float(np.std(past_ret, ddof=1))
        vix_mean_train  = np.mean(past_vix_lagged)
        vix_ratio_train = past_vix_lagged / vix_mean_train

        def objective_function(beta):
            sigma_adj = sigma_train * (vix_ratio_train ** beta[0])
            sigma_adj = np.clip(sigma_adj, 1e-4, 100.0)
            nll = np.sum(np.log(sigma_adj) + 0.5 * ((past_ret - mu_train) / sigma_adj) ** 2)
            return 1e9 if (np.isnan(nll) or np.isinf(nll)) else nll

        maxiter = 200 if cold_start else 30
        cold_start = False

        res = minimize(
            objective_function,
            x0=[last_optimal_beta],
            bounds=[(-0.5, 3.0)],
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'disp': False},
        )
        optimal_beta      = res.x[0]
        last_optimal_beta = optimal_beta

        # Forecast tomorrow
        vix_ratio_pred = vix_array[idx - 1] / vix_mean_train
        sigma_pred     = sigma_train * (vix_ratio_pred ** optimal_beta)
        sigma_pred     = np.clip(sigma_pred, 1e-4, 100.0)

        predictions[idx] = norm.rvs(loc=mu_train, scale=sigma_pred, size=n_samples)

        if idx % 100 == 0:
            print(f"Processed Day {idx}/{T} | β = {optimal_beta:.3f}")

    return predictions


def rolling_vix_scaled_student_t(
    returns: Union[pd.Series, np.ndarray],
    vix: Union[pd.Series, np.ndarray],
    window: int = 252,
    n_samples: int = 2000,
) -> np.ndarray:
    """
    VIX-Scaled Student-T — Advanced Fat-Tail + Exogenous Volatility Baseline.

    Combines the fat-tail realism of the Student-t with VIX-driven dynamic
    volatility scaling.  The degrees of freedom (ν) and location (μ) are
    learned from the rolling historical window, while the scale parameter is
    adjusted by a VIX elasticity factor β:

        scale_adj(t) = scale_hist · normalize(VIX(t-1) / mean(VIX))^β

    The factor is normalized so that its historical mean equals 1, preventing
    artificial inflation of the overall scale.  β is re-estimated daily via
    MLE (Nelder-Mead) with a warm-start for speed.

    Args:
        returns:   Historical realized returns. Shape: (T,)
        vix:       Historical VIX closing prices, aligned with `returns`. Shape: (T,)
        window:    Lookback period for MLE and historical baselines.
        n_samples: Number of paths to draw.

    Returns:
        np.ndarray of shape (T, n_samples). Rows 0…window are NaN (warm-up).
    """
    ret_array = np.asarray(returns, dtype=float).flatten()
    vix_array = np.asarray(vix,     dtype=float).flatten()

    T           = len(ret_array)
    predictions = np.full((T, n_samples), np.nan)
    last_optimal_beta = 0.5
    cold_start        = True

    for idx in range(window + 1, T):
        past_ret        = ret_array[idx - window : idx]
        past_vix_lagged = vix_array[idx - window - 1 : idx - 1]

        df_train, loc_train, scale_train = t.fit(past_ret)
        df_train = np.clip(df_train, 2.0, 100.0)
        
        vix_mean_train  = np.mean(past_vix_lagged)
        vix_ratio_train = past_vix_lagged / vix_mean_train

        def objective_function(beta):
            b = beta[0]
            # Manual bounds penalty for Nelder-Mead (no native bounds support)
            if b < -0.5 or b > 5.0:
                return 1e9

            # Factor normalization: keeps overall scale from inflating
            vix_factor = vix_ratio_train ** b
            vix_factor = vix_factor / np.mean(vix_factor)

            s_adj = np.clip(scale_train * vix_factor, 1e-6, 100.0)
            nll   = -np.sum(t.logpdf(past_ret, df=df_train, loc=loc_train, scale=s_adj))
            return 1e9 if (np.isnan(nll) or np.isinf(nll)) else nll

        # Warm-start: escape the zero-trap on first iteration
        start_guess = last_optimal_beta if last_optimal_beta > 0.05 else 0.5
        maxiter     = 200 if cold_start else 30
        cold_start  = False

        try:
            res          = minimize(
                objective_function,
                x0=[start_guess],
                method='Nelder-Mead',
                options={'maxiter': maxiter, 'disp': False},
            )
            optimal_beta = np.clip(res.x[0], -0.5, 5.0)
            last_optimal_beta = optimal_beta
        except Exception:
            optimal_beta = last_optimal_beta

        # Forecast tomorrow with factor normalization applied
        vix_ratio_pred         = vix_array[idx - 1] / vix_mean_train
        pred_factor            = vix_ratio_pred ** optimal_beta
        historical_factor_mean = np.mean(vix_ratio_train ** optimal_beta)
        pred_factor_normalized = pred_factor / historical_factor_mean

        scale_pred = np.clip(scale_train * pred_factor_normalized, 1e-6, 100.0)
        predictions[idx] = t.rvs(df=df_train, loc=loc_train, scale=scale_pred, size=n_samples)

        if idx % 250 == 0:
            print(f"Processed Day {idx}/{T} | β = {optimal_beta:.3f} | df = {df_train:.2f}")

    return predictions
