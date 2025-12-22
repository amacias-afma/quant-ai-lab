
import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

def check_invariants(series, lags=10):
    print("--- Meucci Invariance Check ---")
    
    # 1. Test for Independence (Ljung-Box Test)
    # Null Hypothesis: Data is independently distributed (no autocorrelation)
    lb_test = acorr_ljungbox(series, lags=[lags], return_df=True)
    lb_pvalue = lb_test['lb_pvalue'].values[0]
    
    print(f"1. Independence (Ljung-Box): p-value = {lb_pvalue:.4f}")
    is_independent = lb_pvalue > 0.05
    if is_independent:
        print("   ✅ PASS: Returns appear independent (No linear memory).")
    else:
        print("   ❌ FAIL: Autocorrelation detected (Linear memory exists).")

    # 2. Test for Identical Distribution (Stability / KS Test)
    # We split the data in half and compare the distributions
    split_point = int(len(series) * 0.5)
    part1 = series.iloc[:split_point]
    part2 = series.iloc[split_point:]
    
    ks_stat, ks_pvalue = stats.ks_2samp(part1, part2)
    
    print(f"2. Stability (KS Test):      p-value = {ks_pvalue:.4f}")
    is_stable = ks_pvalue > 0.05
    if is_stable:
        print("   ✅ PASS: Distribution is stable over time.")
    else:
        print("   ❌ FAIL: Structural break detected (Regime change/Clustering).")
        
    return is_independent, is_stable
