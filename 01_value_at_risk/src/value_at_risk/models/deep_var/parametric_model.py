import pandas as pd
import numpy as np
from scipy import stats

INFORMATIVE_PRIORS = ("param", "hist")
CONTROL_PRIORS = ("shuffled", "constmean", "zero")

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
    df_value_at_risk[f'value_at_risk_hist'] = df_value_at_risk['log_ret'].rolling(252).quantile(alpha)
    df_value_at_risk[f'value_at_risk_param'] = df_value_at_risk[f'mean_{rolling}d'] - z0 * df_value_at_risk[f'std_{rolling}d']
    
    return df_value_at_risk

def build_anchor_prior(df, dates, rolling=22, alpha=0.05, anchor_type="param",
                       control_seed=20260819):
    """Anchor prior aligned to ``dates``. Pure pandas/numpy - no torch, so it is testable.

    Real priors:
      ``param``      rolling Normal VaR (mu - z*sigma)
      ``hist``       rolling historical quantile

    Falsification controls (Risk round 2). An L2 pull toward any FIXED target shrinks every
    seed toward the same point, so inter-seed dispersion falls whether or not the target
    carries risk information. These exist to test whether the observed variance reduction is a
    property of the anchor or merely of shrinkage:

      ``shuffled``   the param prior, time-permuted: same marginal distribution and scale, no
                     information about tomorrow's tail. The sharp control - it separates
                     informativeness from magnitude.
      ``constmean``  a single constant at the prior's mean: right order of magnitude, no time
                     variation.
      ``zero``       shrink toward 0: neither informative nor on-scale.

    The permutation uses a FIXED seed: the target must be the same vector for every model
    seed, otherwise the shrinkage target varies across seeds and the control is invalid.
    Controls are validation-only diagnostics and must never be reported as models.
    """
    df_var = priori_value_at_risk(df, rolling=rolling, alpha=alpha)

    if anchor_type in INFORMATIVE_PRIORS:
        column = "value_at_risk_param" if anchor_type == "param" else "value_at_risk_hist"
        return df_var.loc[dates, column].values

    if anchor_type not in CONTROL_PRIORS:
        raise ValueError(
            f"unknown anchor type {anchor_type!r}; "
            f"expected one of {INFORMATIVE_PRIORS + CONTROL_PRIORS}"
        )

    base = df_var.loc[dates, "value_at_risk_param"].values
    if anchor_type == "zero":
        return np.zeros_like(base)
    if anchor_type == "constmean":
        return np.full_like(base, np.nanmean(base))
    rng = np.random.default_rng(control_seed)
    out = base.copy()
    finite = np.isfinite(out)
    vals = out[finite]
    rng.shuffle(vals)
    out[finite] = vals
    return out
