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
from scipy.optimize import minimize

from arch import arch_model
import warnings

# Suppress arch convergence warnings for clean notebook output
warnings.filterwarnings('ignore', category=RuntimeWarning)

def rolling_garch(returns: pd.Series, window: int = 252, n_samples: int = 2000) -> np.ndarray:
    """
    Fits a GARCH(1,1) model on a rolling window to forecast tomorrow's distribution.
    Assumes a Constant Mean and Normal innovations.
    
    Args:
        returns: Historical realized returns (scaled by 100 is recommended for GARCH).
        window: Lookback period for fitting the GARCH parameters.
        n_samples: Number of paths to draw for the forecasted distribution.
        
    Returns:
        np.ndarray of shape (T, n_samples)
    """
    T = len(returns)
    predictions = np.full((T, n_samples), np.nan)
    
    # GARCH optimization works best when returns are scaled (e.g., percentages like 1.5 instead of 0.015)
    # We assume your data loader already did this, but we keep rescale=False to respect your inputs.
    
    for idx in range(window, T):
        past_window = returns.iloc[idx - window : idx]
        
        # Define the GARCH(1,1) model
        am = arch_model(past_window, vol='Garch', p=1, q=1, mean='Constant', dist='Normal', rescale=False)
        
        try:
            # Fit the model silently
            res = am.fit(disp='off', show_warning=False)
            
            # Forecast tomorrow (horizon=1)
            forecasts = res.forecast(horizon=1, reindex=False)
            mu_pred = forecasts.mean.iloc[-1, 0]
            var_pred = forecasts.variance.iloc[-1, 0]
            sigma_pred = np.sqrt(var_pred)
            
            # Draw samples for our non-parametric evaluation engine
            predictions[idx] = np.random.normal(loc=mu_pred, scale=sigma_pred, size=n_samples)
            
        except Exception:
            # Fallback to simple rolling std if GARCH fails to converge on a specific weird window
            mu_pred = np.mean(past_window)
            sigma_pred = np.std(past_window, ddof=1)
            predictions[idx] = np.random.normal(loc=mu_pred, scale=sigma_pred, size=n_samples)
            
        # Optional: Print progress every 500 days so you know it hasn't frozen
        if idx % 500 == 0:
            print(f"GARCH processing: Day {idx}/{T}")
            
    return predictions

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

# def rolling_vix_gaussian(returns: Union[pd.Series, np.ndarray], 
#                          vix: Union[pd.Series, np.ndarray], 
#                          window: int = 252, 
#                          n_samples: int = 2000) -> np.ndarray:
#     """
#     VIX-Adjusted Gaussian Baseline.
#     Uses the rolling historical mean for drift, but replaces the historical 
#     standard deviation with the VIX-implied daily volatility.
#     """
#     ret_array = returns.values if isinstance(returns, pd.Series) else returns
#     vix_array = vix.values if isinstance(vix, pd.Series) else vix
    
#     T = len(ret_array)
#     predictions = np.full((T, n_samples), np.nan)
    
#     for idx in range(window, T):
#         # Historical window for the mean (drift)
#         past_window = ret_array[idx - window : idx]
#         mu = np.mean(past_window)
        
#         # Exogenous forward-looking volatility from yesterday's VIX
#         # Assuming returns are scaled by 100 (e.g., 1.5%) and VIX is raw (e.g., 20.0)
#         implied_daily_sigma = vix_array[idx - 1] / np.sqrt(252)
        
#         # Draw samples
#         predictions[idx] = norm.rvs(loc=mu, scale=implied_daily_sigma, size=n_samples)
        
#     return predictions


# def rolling_vix_student_t(returns: Union[pd.Series, np.ndarray], 
#                           vix: Union[pd.Series, np.ndarray], 
#                           window: int = 252, 
#                           n_samples: int = 2000) -> np.ndarray:
#     """
#     VIX-Adjusted Student-T Baseline.
#     Learns the 'Fat Tails' (degrees of freedom) and mean from the historical window,
#     but forces the scale parameter to match the VIX-implied volatility.
#     """
#     ret_array = returns.values if isinstance(returns, pd.Series) else returns
#     vix_array = vix.values if isinstance(vix, pd.Series) else vix
    
#     T = len(ret_array)
#     predictions = np.full((T, n_samples), np.nan)
    
#     for idx in range(window, T):
#         past_window = ret_array[idx - window : idx]
        
#         # Fit history to get degrees of freedom (df) and loc (mean)
#         df, loc, _ = t.fit(past_window)
        
#         # We cap df at a minimum of 2.1 so the variance mathematically exists
#         df = max(df, 2.1)
        
#         # Calculate implied daily vol
#         implied_daily_sigma = vix_array[idx - 1] / np.sqrt(252)
        
#         # Adjust the scale parameter. 
#         # For a Student-t, Variance = scale^2 * [df / (df - 2)]
#         # Therefore, scale = sigma * sqrt((df - 2) / df)
#         adjusted_scale = implied_daily_sigma * np.sqrt((df - 2) / df)
        
#         predictions[idx] = t.rvs(df=df, loc=loc, scale=adjusted_scale, size=n_samples)
        
#     return predictions

# def rolling_vix_scaled_student_t(returns: Union[pd.Series, np.ndarray], 
#                                  vix: Union[pd.Series, np.ndarray], 
#                                  window: int = 252, 
#                                  n_samples: int = 2000) -> np.ndarray:
#     """
#     Advanced Baseline: VIX-Scaled Parametric Student-t.
    
#     Captures both 'Fat Tails' (via degrees of freedom) and dynamic volatility scaling 
#     (via VIX elasticity beta).
#     """
#     ret_array = np.asarray(returns, dtype=float).flatten()
#     vix_array = np.asarray(vix, dtype=float).flatten()

#     # ret_array = returns.values if isinstance(returns, pd.Series) else returns
#     # vix_array = np.asarray(vix.values if isinstance(vix, pd.Series) else vix).ravel()  # ensure 1-D
    
#     # WARM START: Keep track of the last optimal beta to speed up the next day's search
#     last_optimal_beta = 0.3

#     T = len(ret_array)
#     predictions = np.full((T, n_samples), np.nan)
#     cold_start = True
#     for idx in range(window + 1, T):
#         print(f'***********{idx}*********************')
#         past_ret = ret_array[idx - window : idx]
#         past_vix_lagged = vix_array[idx - window - 1 : idx - 1]
#         # sigma_train = np.std(past_ret, ddof=1)
        
#         # 1. Fit historical Student-t to get base parameters
#         # df = degrees of freedom, loc = mean, scale = base dispersion
#         df_train, loc_train, scale_train = t.fit(past_ret)
        
#         vix_mean_train = np.mean(past_vix_lagged)
#         vix_ratio_train = past_vix_lagged / vix_mean_train
        
#         # 2. Objective Function: Negative Log-Likelihood for Student-t
#         def objective_function(beta):
#             b = beta[0]
#             # Scale the 'scale' parameter by the VIX ratio raised to beta
#             s_adj = scale_train * (vix_ratio_train ** b)

#             # Clip for numerical safety (no zero or negative scales)
#             s_adj = np.clip(s_adj, 1e-8, None)

#             # Student-t log-likelihood (correct objective for this model)
#             ll = t.logpdf(past_ret, df=df_train, loc=loc_train, scale=s_adj)
#             nll = -np.sum(ll)

#             # Guard against NaN/Inf (can happen at extreme beta values)
#             if np.isnan(nll) or np.isinf(nll):
#                 return 1e9

#             return nll

#         if cold_start:
#             maxiter = 100
#             cold_start = False
#         else:
#             maxiter = 20
        
        
#         # 4. THE CALLBACK: This prints the optimizer's internal steps
#         iteration_counter = [0] # Use a list so it can be updated inside the function

#         def debug_callback(current_guess):
#             iteration_counter[0] += 1
#             current_beta = current_guess[0]
#             current_nll = objective_function(current_guess)
#             print(f"   [Iteration {iteration_counter[0]:02d}] Tested Beta: {current_beta:>7.4f} | Resulting NLL: {current_nll:.4f}")
        
#         # 3. Optimize Beta
#         res = minimize(
#                 objective_function, 
#                 x0=[last_optimal_beta], 
#                 bounds=[(-1.0, 5.0)], 
#                 # method='L-BFGS-B',
#                 method='Nelder-Mead',
#                 callback=debug_callback,
#                 options={'maxiter': maxiter, 'disp': False}
#                 # options={'maxiter': maxiter, 'ftol': 1e-4, 'disp': False}
#             )
#         # res = minimize(objective_function, x0=[0.0], bounds=[(-1.0, 5.0)], method='L-BFGS-B')
#         optimal_beta = res.x[0]
#         last_optimal_beta = optimal_beta # Save for tomorrow's warm start
        
#         # 4. Forecast Tomorrow
#         vix_ratio_pred = vix_array[idx - 1] / vix_mean_train
#         scale_pred = scale_train * (vix_ratio_pred ** optimal_beta)
        
#         # Draw samples for evaluation
#         predictions[idx] = t.rvs(df=df_train, loc=loc_train, scale=scale_pred, size=n_samples)
#         # Progress tracker so you know it hasn't frozen
#         if idx % 10 == 0:
#             print(f"Processed Day {idx}/{T} | Current optimal beta: {optimal_beta:.3f}")
#     return predictions

# import numpy as np
# import pandas as pd
# from scipy.stats import t
# from scipy.optimize import minimize
# from typing import Union

def rolling_vix_scaled_student_t(returns: Union[pd.Series, np.ndarray], 
                                 vix: Union[pd.Series, np.ndarray], 
                                 window: int = 252, 
                                 n_samples: int = 2000) -> np.ndarray:
    """
    Advanced Baseline: VIX-Scaled Parametric Student-t.
    """
    ret_array = np.asarray(returns, dtype=float).flatten()
    vix_array = np.asarray(vix, dtype=float).flatten()
    
    T = len(ret_array)
    predictions = np.full((T, n_samples), np.nan)
    
    # WARM START
    last_optimal_beta = 0.5
    cold_start = True
    
    for idx in range(window + 1, T):
        past_ret = ret_array[idx - window : idx]
        past_vix_lagged = vix_array[idx - window - 1 : idx - 1]
        
        # 1. Fit historical Student-t 
        # (This is slow, but mathematically required for the baseline)
        df_train, loc_train, scale_train = t.fit(past_ret)
        
        vix_mean_train = np.mean(past_vix_lagged)
        vix_ratio_train = past_vix_lagged / vix_mean_train
        
        # 2. Robust Objective Function
        def objective_function(beta):
            b = beta[0]
            
            # MANUAL BOUNDS FOR NELDER-MEAD: 
            # If the optimizer guesses a crazy number, penalize it heavily
            if b < -0.5 or b > 5.0:
                return 1e9
                
            # THE SECRET SAUCE: Factor Normalization
            # We normalize the factor so its mean is ALWAYS 1.0. 
            # This prevents the overall scale from artificially inflating.
            vix_factor = vix_ratio_train ** b
            vix_factor = vix_factor / np.mean(vix_factor) 
            
            s_adj = scale_train * vix_factor

            # Clip for numerical safety
            s_adj = np.clip(s_adj, 1e-6, 100.0)

            # Student-t Negative Log-Likelihood
            ll = t.logpdf(past_ret, df=df_train, loc=loc_train, scale=s_adj)
            nll = -np.sum(ll)

            if np.isnan(nll) or np.isinf(nll):
                return 1e9

            return nll

        # 3. Escape the Zero-Trap
        start_guess = last_optimal_beta if last_optimal_beta > 0.05 else 0.5
        if cold_start:
            maxiter = 200
            cold_start = False
        else:
            maxiter = 30
        
        # 4. Optimize Beta (Removed 'bounds' parameter, relying on manual penalty)
        try:
            res = minimize(
                objective_function, 
                x0=[start_guess], 
                method='Nelder-Mead',
                options={'maxiter': maxiter, 'disp': False}
            )
            optimal_beta = res.x[0]
            # Clip the final result just in case
            optimal_beta = np.clip(optimal_beta, -0.5, 5.0)
            last_optimal_beta = optimal_beta 
            
        except Exception:
            optimal_beta = last_optimal_beta
        
        # 5. Forecast Tomorrow
        vix_ratio_pred = vix_array[idx - 1] / vix_mean_train
        
        # We must apply the same logic to the prediction!
        pred_factor = vix_ratio_pred ** optimal_beta
        # Approximate normalization using the historical mean of the factor
        historical_factor_mean = np.mean(vix_ratio_train ** optimal_beta)
        pred_factor_normalized = pred_factor / historical_factor_mean
        
        scale_pred = scale_train * pred_factor_normalized
        scale_pred = np.clip(scale_pred, 1e-6, 100.0)
        
        # Draw samples
        predictions[idx] = t.rvs(df=df_train, loc=loc_train, scale=scale_pred, size=n_samples)
        
        # Progress tracker
        if idx % 250 == 0:
            print(f"Processed Day {idx}/{T} | Current optimal beta: {optimal_beta:.3f} | df: {df_train:.2f}")

    return predictions

def rolling_vix_scaled_gaussian(returns: Union[pd.Series, np.ndarray], 
                                vix: Union[pd.Series, np.ndarray], 
                                window: int = 252, 
                                n_samples: int = 2000) -> np.ndarray:
    
    # Forcefully convert to 1D numpy arrays to prevent Pandas broadcasting errors
    ret_array = np.asarray(returns, dtype=float).flatten()
    vix_array = np.asarray(vix, dtype=float).flatten()

    # ret_array = returns.values if isinstance(returns, pd.Series) else returns

    # vix_array = vix.values if isinstance(vix, pd.Series) else vix
    
    T = len(ret_array)
    predictions = np.full((T, n_samples), np.nan)
    
    # WARM START: Keep track of the last optimal beta to speed up the next day's search
    last_optimal_beta = 0.3
    
    # We add a print statement counter to monitor progress
    print(f"Starting VIX optimization for {T - window} days...")
    cold_start = True

    for idx in range(window + 1, T):
        past_ret = ret_array[idx - window : idx]
        past_vix_lagged = vix_array[idx - window - 1 : idx - 1]
        
        mu_train = float(np.mean(past_ret))
        sigma_train = float(np.std(past_ret, ddof=1))

        vix_mean_train = np.mean(past_vix_lagged)
        
        vix_ratio_train = past_vix_lagged / vix_mean_train
        
        # --- ROBUST OBJECTIVE FUNCTION ---
        def objective_function(beta):
            b = beta[0]
            # 1. Calculate adjusted sigma
            sigma_adj = sigma_train * (vix_ratio_train ** b)
            
            # 2. NUMERICAL SAFETY: Clip sigma to prevent division by zero or log(0)
            sigma_adj = np.clip(sigma_adj, 1e-4, 100.0)
            
            # 3. Manual NLL calculation is often faster and safer than scipy.stats inside a loop
            # NLL = sum( ln(sigma) + 0.5 * ((x - mu) / sigma)^2 )
            nll = np.sum(np.log(sigma_adj) + 0.5 * ((past_ret - mu_train) / sigma_adj)**2)
            
            # 4. If NLL blows up, return a massive penalty to push the optimizer back
            if np.isnan(nll) or np.isinf(nll):
                return 1e9
                
            return nll
        
        # --- ROBUST OPTIMIZER SETTINGS ---
        # x0 uses the Warm Start. 
        # bounds prevent extreme exponents. 
        # options constrain the maximum iterations so it NEVER gets stuck for hours.
        # try:
        if cold_start:
            maxiter = 200
            cold_start = False
        else:
            maxiter = 30

        # 4. THE CALLBACK: This prints the optimizer's internal steps
        iteration_counter = [0] # Use a list so it can be updated inside the function

        # def debug_callback(current_guess):
        #     iteration_counter[0] += 1
        #     current_beta = current_guess[0]
        #     current_nll = objective_function(current_guess)
        #     print(f"   [Iteration {iteration_counter[0]:02d}] Tested Beta: {current_beta:>7.4f} | Resulting NLL: {current_nll:.4f}")
        
        res = minimize(
            objective_function, 
            x0=[last_optimal_beta], 
            bounds=[(-0.5, 3.0)], 
            # method='L-BFGS-B',
            # callback=debug_callback,

            method='Nelder-Mead',
            # options={'maxiter': maxiter, 'ftol': 1e-4, 'disp': False}
            options={'maxiter': maxiter, 'disp': False}

        )
        optimal_beta = res.x[0]
        last_optimal_beta = optimal_beta # Save for tomorrow's warm start
            
        # except Exception:
        #     # Fallback mechanism: If the optimizer mathematically crashes, just use yesterday's beta
        #     optimal_beta = last_optimal_beta
        
        # Forecast Tomorrow
        mu_pred = mu_train 
        vix_ratio_pred = vix_array[idx - 1] / vix_mean_train
        
        # Safety clip on the final prediction
        sigma_pred = sigma_train * (vix_ratio_pred ** optimal_beta)
        sigma_pred = np.clip(sigma_pred, 1e-4, 100.0)
        
        predictions[idx] = norm.rvs(loc=mu_pred, scale=sigma_pred, size=n_samples)
        
        # Progress tracker so you know it hasn't frozen
        if idx % 100 == 0:
            print(f"Processed Day {idx}/{T} | Current optimal beta: {optimal_beta:.3f}")
            
    return predictions

# def rolling_vix_scaled_gaussian(returns: Union[pd.Series, np.ndarray], 
#                                 vix: Union[pd.Series, np.ndarray], 
#                                 window: int = 252, 
#                                 n_samples: int = 2000) -> np.ndarray:
#     """
#     Advanced Baseline: VIX-Scaled Parametric Gaussian.
    
#     Dynamically learns the elasticity parameter (beta) between historical volatility 
#     and the exogenous VIX index using Maximum Likelihood Estimation (MLE).
#     It adjusts the standard deviation for tomorrow's forecast based on today's VIX.
    
#     Args:
#         returns: Historical realized returns (Shape: T,)
#         vix: Historical VIX closing prices (Shape: T, parallel to returns)
#         window: Lookback period for MLE optimization and historical baseline.
#         n_samples: Number of paths to draw for the forecasted distribution.
        
#     Returns:
#         np.ndarray of shape (T, n_samples)
#     """
#     ret_array = returns.values if isinstance(returns, pd.Series) else returns
#     vix_array = vix.values if isinstance(vix, pd.Series) else vix
#     vix_array = np.asarray(vix_array).ravel()  # ensure 1-D — df.values gives (T,1) which breaks slicing inside logpdf
    
#     T = len(ret_array)
#     predictions = np.full((T, n_samples), np.nan)
    
#     # We start at window + 1 to ensure we have a full window of LAGGED VIX data
#     for idx in range(window + 1, T):
        
#         # 1. Isolate the training data (Strictly past data to prevent look-ahead bias)
#         # We want to predict past_ret using the VIX from the day BEFORE each return
#         past_ret = ret_array[idx - window : idx]
#         past_vix_lagged = vix_array[idx - window - 1 : idx - 1]
        
#         # 2. Compute historical baselines for the training window
#         mu_train = np.mean(past_ret)
#         sigma_train = np.std(past_ret, ddof=1)
#         vix_mean_train = np.mean(past_vix_lagged)
        
#         # Ratio of VIX relative to its recent average
#         vix_ratio_train = past_vix_lagged / vix_mean_train
        
#         # 3. Define the Negative Log-Likelihood (NLL) Objective Function
#         def objective_function(beta):
#             # Scale historical volatility using the VIX ratio and beta
#             sigma_adj = sigma_train * (vix_ratio_train ** beta)
            
#             # Calculate exact Log-Likelihood of the training window
#             ll = norm.logpdf(past_ret, loc=mu_train, scale=sigma_adj + 1e-8)
#             return -np.sum(ll)
        
#         # 4. Optimize Beta using L-BFGS-B (Bounded optimization)
#         # Beta > 0 means volatility expands when VIX rises.
#         res = minimize(objective_function, x0=[0.0], bounds=[(-1.0, 5.0)], method='L-BFGS-B')
#         optimal_beta = res.x[0]
        
#         # 5. Forecast Tomorrow's Distribution
#         # Use yesterday's VIX (idx - 1) to predict today (idx)
#         mu_pred = mu_train 
#         vix_ratio_pred = vix_array[idx - 1] / vix_mean_train
        
#         # Apply the learned optimal beta to forecast tomorrow's volatility
#         sigma_pred = sigma_train * (vix_ratio_pred ** optimal_beta)
        
#         # Draw samples for the universal evaluation engine
#         predictions[idx] = norm.rvs(loc=mu_pred, scale=sigma_pred, size=n_samples)
        
#     return predictions
