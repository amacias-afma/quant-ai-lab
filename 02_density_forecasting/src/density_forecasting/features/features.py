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

    df_features = df.copy()
    df_features['returns'] = 100 * np.log(df_features['prices']).diff()
    # df_features = 100 * df[['returns']].copy()
    # df_features =
    df_features['returns_m1'] = df_features['returns'].shift(1)
    df_features['returns_m2'] = df_features['returns'].shift(2)
    df_features['returns_m3'] = df_features['returns'].shift(3)

    df_features['returns_m1_2'] = df_features['returns_m1']**2
    df_features['returns_m2_2'] = df_features['returns_m2']**2
    df_features['returns_m3_2'] = df_features['returns_m3']**2

    if 'volume' in df.columns:
        df_features['volume'] = df['volume'].replace(0, np.nan).ffill().bfill().shift()
        
        # Deep Features
        df_features['Vol_50_MA'] = df_features['volume'].rolling(window=50).mean()
        df_features['Feat_Volume_Shock'] = df_features['volume'] / df_features['Vol_50_MA']

    df_features['Feat_Momentum_20'] = df_features['returns_m1'].rolling(window=20).mean()
    df_features['Feat_Realized_Vol_20'] = df_features['returns_m1'].rolling(window=20).std()

    # RSI
    delta = df['prices'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df_features['Feat_RSI_14'] = 100 - (100 / (1 + (gain / loss)))

    df_features['returns_m1_mean'] = df_features['returns_m1'].rolling(window_size).mean()
    df_features['returns_m1_std'] = df_features['returns_m1'].rolling(window_size).std()
    df_features['returns_m1_kurt'] = df_features['returns_m1'].rolling(window_size).kurt()
    df_features['returns_m1_skew'] = df_features['returns_m1'].rolling(window_size).skew()

    df_features['returns_m1_mean_st'] = df_features['returns_m1'].rolling(11).mean()
    df_features['returns_m1_std_st'] = df_features['returns_m1'].rolling(11).std()

    df_features['returns_m1_nu'] = 4 + (6 / df_features['returns_m1_kurt'])
    df_features.loc[df_features['returns_m1_nu'] > 30, 'returns_m1_nu'] = 30
    df_features.loc[df_features['returns_m1_nu'] < 2, 'returns_m1_nu'] = 2

    df_features = pd.concat([df_features, df_vix], axis=1)
    df_features.ffill(inplace=True)
    df_features['VIX'] = df_features['VIX'].shift()       # no look-ahead

    df_features['vix_ratio'] = df_features['VIX'] / df_features['VIX'].rolling(window_size).mean()


    return df_features
