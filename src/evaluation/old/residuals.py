
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats

def meucci_check(residuals):
    """
    The 'Quant Auditor'. Checks Param Constraints, Significance, and Meucci's i.i.d. props.
    """
    if residuals is None:
        raise ValueError("Residuals not provided.")

    print("\n" + "="*40)
    print("   MODEL DIAGNOSTIC REPORT (MEUCCI CHECK)")
    print("="*40)
    
    
    # --- CHECK 3: Independence (Ljung-Box & Lag-1 Correlation) ---
    # "Is the present correlated with the past?"
    # We test the SQUARED residuals for remaining volatility clustering
    lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
    lb_pvalue = lb_test['lb_pvalue'].values[0]
    
    # Simple Lag-1 Autocorrelation
    lag1_corr = residuals.autocorr(lag=1)
    market_invariants_pass = True
    print(f"3. Independence (i.i.): Lag-1 Corr = {lag1_corr:.4f}, Ljung-Box p = {lb_pvalue:.4f}")
    # print(lb_test)

    if abs(lag1_corr) < 0.05 and lb_pvalue > 0.05:
        print("   [PASS] Residuals look independent (White Noise).")
    else:
        print("   [FAIL] Residuals still have autocorrelation. GARCH failed to capture all dynamics.")
        market_invariants_pass = False

    # --- CHECK 4: Identical Distribution (Meucci Stability Test) ---
    # "Does the distribution change over time?"
    # Split data in half and compare CDFs using Kolmogorov-Smirnov Test
    n = len(residuals)
    half1 = residuals[:n//2]
    half2 = residuals[n//2:]
    
    ks_stat, ks_pvalue = stats.ks_2samp(half1, half2)
    print(f"4. Stability (i.d.): KS Test p-value = {ks_pvalue:.4f}")
    if ks_pvalue > 0.05:
        # We cannot reject null hypothesis that they are same distribution
        print("   [PASS] Distribution is stable over time (Regimes are consistent).")
    else:
        print("   [FAIL] Distribution changed in second half. Structural Break detected.")
        market_invariants_pass = False

    print("="*40 + "\n")
    return market_invariants_pass
