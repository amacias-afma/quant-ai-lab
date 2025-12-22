from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
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

def garch_fit(modeling_series, is_stable_resid):
    # Step 6: Instability / Volatility Clustering Check
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