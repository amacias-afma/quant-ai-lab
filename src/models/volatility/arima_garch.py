from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from scipy.stats import norm
import pandas as pd
import numpy as np
    
from statsmodels.stats.diagnostic import acorr_ljungbox
from src.utils.invariants import check_invariants
import matplotlib.pyplot as plt

# Modeling Decision Tree based on Test Results
def arima_fit(df_clean, is_independent, is_stable):
        
    if not is_independent:
        print("\n>>> Detected Linear Dependence. Fitting ARIMA Model...")
        
        # Step 5: Fit ARIMA (Auto-Regressive Integrated Moving Average)
        # We use (1,0,1) as a starting point, but you could use auto_arima to find optimal p,d,q
        arima_model = ARIMA(df_clean['log_ret'], order=(1, 0, 1))
        arima_result = arima_model.fit()
        
        print(arima_result.summary())
        
        # Extract Residuals from ARIMA for the next check
        residuals = arima_result.resid
        
        # Check if ARIMA fixed the independence issue
        lb_p = acorr_ljungbox(residuals, lags=[10], return_df=True)['lb_pvalue'].values[0]
        if lb_p > 0.05:
            print("   ✅ ARIMA successfully removed linear dependence.")
        
        # Now we check these residuals for stability/volatility clustering
        print("   Checking ARIMA residuals for stability...")
        _, is_stable_resid = check_invariants(residuals)
        
        # Prepare data for next step (if needed)
        modeling_series = residuals

    else:
        print("\n>>> No Linear Dependence. Proceeding with raw returns...")
        modeling_series = df_clean['log_ret']
        # We carry over the stability result from the first test
        is_stable_resid = is_stable
        
    return modeling_series, is_stable_resid


# Step 6: Instability / Volatility Clustering Check
def garch_fit(modeling_series, is_stable_resid):
    if not is_stable_resid:
        print("\n>>> Detected Distribution Instability (Volatility Clustering). Fitting GARCH Model...")
        
        # TRICK: Scale returns by 100 for GARCH numerical stability (as seen in baseline.py)
        scaled_series = modeling_series * 100
        
        # Fit GARCH(1,1)
        garch_model = arch_model(scaled_series, vol='Garch', p=1, o=0, q=1, dist='Normal')
        garch_result = garch_model.fit(disp='off')
        
        print(garch_result.summary())
        
        # Final Check: Are the GARCH standardized residuals invariants?
        print("\n>>> Final Check: GARCH Standardized Residuals")
        std_resid = garch_result.std_resid
        check_invariants(std_resid)
        
        # Visualize the Volatility
        garch_result.conditional_volatility.plot(title="GARCH Estimated Volatility (Scaled)")
        plt.show()

    else:
        print("\n>>> Data appears to be stationary random noise (White Noise). No GARCH needed.")

def arima_garch_forecast(train_data, test_data, p=1, q=1, garch_p=1, garch_q=1, confidence_level=0.99):
    """
    Forecasting VaR using ARIMA-GARCH on TEST data (Out-of-Sample).
    
    Args:
        train_data (pd.Series): Scaled training log-returns.
        test_data (pd.Series): Scaled test log-returns.
        p, q: ARIMA order (d=0 assumed for log-returns).
        garch_p, garch_q: GARCH order.
        confidence_level: VaR confidence (default 0.99).
        
    Returns:
        pd.Series: Forecasted VaR for the test period.
    """

    print(f"\n⚡ Fitting ARIMA({p},0,{q})-GARCH({garch_p},{garch_q}) for VaR Projection...")
    
    # 1. Fit ARIMA on Train
    arima_model = ARIMA(train_data, order=(p, 0, q))
    arima_res = arima_model.fit()
    
    # 2. Forecast Mean (ARIMA) for Test Period
    # We use 'apply' to run the filter over test data with fixed parameters (mimicking real-time usage without full retrain)
    # Alternatively one could re-fit, but that's slow. 'apply' is standard for backtesting.
    arima_test_res = arima_res.apply(test_data)
    mean_forecast = arima_test_res.fittedvalues # Expected return at t
    
    # 3. Get Residuals for GARCH
    train_residuals = arima_res.resid
    
    # 4. Fit GARCH on Train Residuals
    # Scale is already handled (input data should be scaled x100)
    garch_model = arch_model(train_residuals, vol='Garch', p=garch_p, o=0, q=garch_q, dist='Normal')
    garch_res = garch_model.fit(disp='off')
    
    # 5. Forecast Volatility (GARCH) for Test Period
    # We apply the GARCH model to the TEST residuals from ARIMA
    # Test residuals = Actual Test Data - ARIMA Mean Forecast
    test_residuals = test_data - mean_forecast
    
    # Fix GARCH params and filter over test residuals
    # We must create a NEW model instance bound to the test residuals
    garch_model_test = arch_model(test_residuals, vol='Garch', p=garch_p, o=0, q=garch_q, dist='Normal')
    garch_test_res = garch_model_test.fix(garch_res.params)
    vol_forecast = garch_test_res.conditional_volatility
    
    # 6. Calculate VaR
    # VaR = Mean_Forecast - (Z * Vol_Forecast)
    # Since we are modeling returns, a loss is negative return.
    # VaR_99 is the threshold where P(r < VaR) = 0.01
    
    z_score = norm.ppf(1 - confidence_level) # e.g. -2.33 for 99%
    # VaR is typically expressed as a positive capital requirement:
    # "Loss will not exceed X". So if return is -2%, and VaR is 3%, we are safe.
    # Formula: VaR = -(Mean + Z * Sigma)
    
    var_forecast = -(mean_forecast + z_score * vol_forecast)
    
    # Align index
    var_forecast.index = test_data.index
    return var_forecast