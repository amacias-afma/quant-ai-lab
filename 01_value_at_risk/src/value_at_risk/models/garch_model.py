import pandas as pd
import numpy as np
import datetime as dt
from scipy import stats

# NOTE: ``arch`` is imported lazily inside calculate_garch_var so that the pure-math helper
# (standardized_t_quantile) and its tests do not require the arch package to be installed.


def standardized_t_quantile(alpha, nu):
    """Quantile at level ``alpha`` of a Student-t standardized to unit variance.

    arch fits GARCH innovations as a *standardized* Student-t (unit variance). scipy's
    ``stats.t.ppf`` returns the quantile of the *unstandardized* t, whose variance is
    ``nu / (nu - 2)``. To put it on unit variance we MULTIPLY by ``sqrt((nu - 2) / nu)``.

    The previous implementation divided by that factor instead, inflating the tail quantile
    by ``nu / (nu - 2)`` and blowing up VaR for fat-tailed series (small nu) — e.g. the
    BTC "-443% capital reserved" artefact. ``nu`` is clamped just above 2, where the
    variance (and hence the standardization) is defined.
    """
    nu = float(nu)
    if nu <= 2.0 + 1e-6:
        nu = 2.0 + 1e-6
    return stats.t.ppf(alpha, df=nu) * np.sqrt((nu - 2.0) / nu)


def calculate_garch_var(returns, split_date=None, window=132, alpha=0.05):
    """
    Calculate 1-step ahead GARCH(1,1) VaR using Student's T distribution.
    
    If split_date is provided, the GARCH parameters are fit only every 22 days,
    and out-of-sample forecasting is performed day-by-day using fixed parameters
    but updating the daily volatility filter with newly arriving returns.
    
    If split_date is None, a rolling window GARCH model is computed for all steps.
    """
    from arch import arch_model  # lazy: keeps the module importable without arch

    var_garch = pd.Series(index=returns.index, dtype=float)
    # Fallback normal quantile if T estimation fails
    z_alpha_norm = abs(stats.norm.ppf(alpha))
    
    if split_date is not None:
        split_dt = pd.to_datetime(split_date)
        # Find index of split_date
        start_idx = returns.index.get_indexer([split_dt], method='pad')[0]
        
        idx = start_idx
        while idx < len(returns) - 1:
            end_idx = min(idx + 23, len(returns))
            out_of_sample_dates = returns.index[idx + 1 : end_idx]
            
            # Fit GARCH model using data up to index 'idx' (expanding window)
            train_window = returns.iloc[:idx + 1]
            rescaled_train = train_window * 100
            
            try:
                am = arch_model(rescaled_train, vol='Garch', p=1, q=1, dist='studentst', rescale=False)
                res = am.fit(disp='off')
                params = res.params
                
                # Student's t quantile (standardized to unit variance, as arch fits it)
                nu = params.get('nu', 5.0) # default to 5 if not found
                z_alpha_t = abs(standardized_t_quantile(alpha, nu))
                
                # Update volatility filter with the out-of-sample data using fixed parameters
                all_data = returns.iloc[:end_idx] * 100
                am_window = arch_model(all_data, vol='Garch', p=1, q=1, dist='studentst', rescale=False)
                res_fixed = am_window.fix(params)
                
                # Forecast VaR for each day in the out-of-sample window
                for t in out_of_sample_dates:
                    vol_t = res_fixed.conditional_volatility.loc[t] / 100.0
                    mean_t = params['mu'] / 100.0
                    var_garch.loc[t] = mean_t - z_alpha_t * vol_t
                    
            except Exception:
                # Fallback to rolling historical window if GARCH fails
                fallback_window = returns.iloc[max(0, idx + 1 - window):idx + 1]
                rolling_mean = fallback_window.mean()
                rolling_std = fallback_window.std()
                for t in out_of_sample_dates:
                    var_garch.loc[t] = rolling_mean - z_alpha_norm * rolling_std
                    
            idx = end_idx - 1
            
    else:
        # Fallback to the original rolling window calculation for the entire series
        for i in range(window, len(returns)):
            train_window = returns.iloc[i - window:i]
            rescaled_returns = train_window * 100
            try:
                am = arch_model(rescaled_returns, vol='Garch', p=1, q=1, dist='studentst', rescale=False)
                res = am.fit(disp='off')
                forecasts = res.forecast(horizon=1)
                
                nu = res.params.get('nu', 5.0)
                z_alpha_t = abs(standardized_t_quantile(alpha, nu))

                forecast_var = forecasts.variance.iloc[-1, 0] / 10000.0
                forecast_mean = forecasts.mean.iloc[-1, 0] / 100.0
                var_garch.iloc[i] = forecast_mean - z_alpha_t * np.sqrt(forecast_var)
            except Exception:
                rolling_mean = train_window.mean()
                rolling_std = train_window.std()
                var_garch.iloc[i] = rolling_mean - z_alpha_norm * rolling_std
                
    return var_garch.dropna()
    
