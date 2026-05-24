"""
Feature Engineering for Density Forecasting.

This module provides utilities to build the feature matrix used by the
Student-t neural network models. Features include:
  - Lagged returns (momentum signals)
  - Rolling Student-t distribution parameters (nu, mu, sigma) as
    statistical priors for the residual network
  - Externally provided volatility signals such as VIX
"""

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_cache: dict = {}


def _fit_and_get(idx: np.ndarray) -> tuple:
    """Fit a Student-t once per unique window slice and cache the result.

    This is a small internal memoization helper to avoid refitting the same
    data repeatedly across overlapping calls.

    Args:
        idx: 1-D NumPy array of return observations for one rolling window.

    Returns:
        Tuple (nu, mu, sigma) — the MLE parameters of the Student-t fit.
    """
    key = tuple(idx)
    if key not in _cache:
        _cache[key] = stats.t.fit(idx)
    return _cache[key]


# ---------------------------------------------------------------------------
# Public feature constructors
# ---------------------------------------------------------------------------

def rolling_t_fit(df: pd.DataFrame, col: str, window: int) -> pd.DataFrame:
    """Fit a Student-t distribution to every rolling window of a column.

    For each position i in [window, len(df)], the function fits a
    Student-t distribution via MLE to the slice df[col][i-window:i] and
    stores the resulting (nu, mu, sigma) parameters.

    Args:
        df:     Input DataFrame whose index will be preserved in the output.
        col:    Name of the column to fit (e.g. 'ret_1' for lagged returns).
        window: Number of observations in each rolling window.

    Returns:
        pd.DataFrame with columns ['nu', 'mu', 'sigma'] and the same index
        as ``df`` starting from row ``window - 1``.  Rows with any NaN in
        the window are filled with (NaN, NaN, NaN).
    """
    params = []
    arr = df[col].to_numpy()

    for i in range(window, len(arr) + 1):
        window_data = arr[i - window:i]
        if np.isnan(window_data).any():
            params.append((np.nan, np.nan, np.nan))
        else:
            params.append(stats.t.fit(window_data))

    result = pd.DataFrame(
        params,
        index=df.index[window - 1:],
        columns=['nu', 'mu', 'sigma'],
    )
    return result


def features_lags(df_features: pd.DataFrame, n_lags: int = 5) -> None:
    """Add lagged return columns to ``df_features`` in-place.

    Creates columns 'ret_1', 'ret_2', ..., 'ret_n_lags', where 'ret_k' is
    the return shifted k trading days into the past.  The column 'returns'
    must already exist in ``df_features``.

    Args:
        df_features: DataFrame that already contains a 'returns' column.
                     Modified in-place.
        n_lags:      Number of lags to generate.  Defaults to 5.
    """
    for i in range(1, n_lags + 1):
        df_features[f'ret_{i}'] = df_features['returns'].shift(i)


def create_features(
    df: pd.DataFrame,
    df_vix: pd.DataFrame,
    window_size: int,
    n_lags: int=1
) -> pd.DataFrame:
    """Build the full feature matrix for the walk-forward backtest.

    Pipeline:
        1. Scale returns to percentage points (× 100) for numerical stability.
        2. Merge the (already annualisation-adjusted) VIX series and shift it
           by one day to prevent look-ahead bias.
        3. Add ``n_lags=5`` lagged return columns via :func:`features_lags`.
        4. Fit a rolling Student-t to 'ret_1' over ``window_size`` days and
           append (nu, mu, sigma) as additional features.
        5. Cap nu at 30 and normalise it to [0, 1] so the residual network
           receives a bounded input.
        6. Drop all rows with NaN (warm-up rows at the start of the series).

    Args:
        df:          DataFrame returned by ``fetch_asset_data``.  Must contain
                     a 'returns' column with raw (fractional) daily returns.
        df_vix:      Single-column DataFrame with the VIX series aligned to
                     the same calendar as ``df``, already scaled to daily
                     units (e.g. VIX / sqrt(252)).
        window_size: Number of trading days used for the rolling Student-t
                     fit (e.g. 3 * 22 for a 3-month window).

    Returns:
        pd.DataFrame with all features and the 'returns' target column.
        Rows containing any NaN (warm-up period) are dropped.
    """
    df_features = 100 * df[['returns']].copy()
    df_features = pd.concat([df_features, df_vix], axis=1)
    df_features.ffill(inplace=True)
    df_features['VIX'] = df_features['VIX'].shift()       # no look-ahead

    features_lags(df_features, n_lags)
    df_features['ret_1_mean'] = df_features['ret_1'].rolling(window_size).mean()
    df_features['ret_1_std'] = df_features['ret_1'].rolling(window_size).std()
    df_features['ret_1_kurt'] = df_features['ret_1'].rolling(window_size).kurt()
    df_features['ret_1_skew'] = df_features['ret_1'].rolling(window_size).skew()

    if 'volume' in df.columns:
        # df_features['volume'] = df['volume'].shift()  # no look-ahead
        df_features['volume_var'] = np.log(df['volume'].shift()).diff() # no look-ahead

    
    # params_df = rolling_t_fit(df_features, 'ret_1', window_size)
    # df_features[['nu', 'mu', 'sigma']] = params_df
    # df_features.loc[df_features['nu'] > 30, 'nu'] = 30    # cap heavy tails
    # df_features['nu'] /= 30                                # normalise to [0, 1]

    # df_features.dropna(inplace=True)
    return df_features
