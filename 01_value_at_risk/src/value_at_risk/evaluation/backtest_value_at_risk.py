import pandas as pd
import numpy as np

def backtest_var_models(df_results, realized_col='realized', alpha=0.05):
    """
    Backtest and compare VaR models.
    
    Parameters:
    -----------
    df_results : pd.DataFrame
        DataFrame with columns: 'realized', 'predicted' (neural network VaR),
        and optionally 'parametric' (parametric VaR)
    realized_col : str
        Column name for realized returns
    alpha : float
        Significance level
    
    Returns:
    --------
    dict : Backtesting statistics for each model
    """
    results = {}
    
    # Models to test
    models = {}
    if 'predicted' in df_results.columns:
        models['Neural Network'] = 'predicted'
    if 'parametric' in df_results.columns:
        models['Parametric (Normal)'] = 'parametric'
    
    for model_name, var_col in models.items():
        # Calculate breaches
        breaches = (df_results[realized_col] < df_results[var_col]).astype(int)
        
        # Calculate statistics
        total_obs = len(breaches)
        total_breaches = breaches.sum()
        breach_rate = total_breaches / total_obs
        expected_breaches = alpha * total_obs
        
        # Store results
        results[model_name] = {
            'total_observations': total_obs,
            'total_breaches': total_breaches,
            'breach_rate': breach_rate,
            'expected_breaches': expected_breaches,
            'expected_rate': alpha,
            'difference': breach_rate - alpha
        }
    
    return results

def var_backtesting_tests(df_results, alpha=0.01, confidence_level=0.95):
    """
    Perform statistical tests to validate VaR model performance.
    
    Parameters:
    -----------
    df_results : DataFrame
        DataFrame with 'predicted' (VaR) and 'realized' (actual returns) columns
    alpha : float
        VaR confidence level (e.g., 0.01 for 99% VaR)
    confidence_level : float
        Statistical test confidence level (typically 0.95)
    
    Returns:
    --------
    dict : Dictionary with test results and interpretation
    """
    from scipy import stats
    
    # Calculate breaches
    breaches = (df_results['realized'] < df_results['predicted']).astype(int)
    n_breaches = breaches.sum()
    n_obs = len(breaches)
    breach_rate = n_breaches / n_obs
    
    results = {
        'n_observations': n_obs,
        'n_breaches': n_breaches,
        'breach_rate': breach_rate,
        'expected_rate': alpha,
        'tests': {}
    }
    
    # ============================================
    # 1. KUPIEC POF TEST (Proportion of Failures)
    # ============================================
    # H0: The breach rate equals the expected rate (alpha)
    # Test statistic follows chi-square distribution with 1 df
    
    if n_breaches == 0 or n_breaches == n_obs:
        pof_stat = np.inf
        pof_pvalue = 0.0
    else:
        pof_stat = -2 * (
            np.log((1 - alpha)**(n_obs - n_breaches) * alpha**n_breaches) -
            np.log((1 - breach_rate)**(n_obs - n_breaches) * breach_rate**n_breaches)
        )
        pof_pvalue = 1 - stats.chi2.cdf(pof_stat, df=1)
    
    pof_critical = stats.chi2.ppf(confidence_level, df=1)
    pof_reject = pof_stat > pof_critical
    
    results['tests']['kupiec_pof'] = {
        'statistic': pof_stat,
        'p_value': pof_pvalue,
        'critical_value': pof_critical,
        'reject_h0': pof_reject,
        'conclusion': 'REJECT - Model is inadequate' if pof_reject else 'ACCEPT - Breach rate is acceptable'
    }
    
    # ============================================
    # 2. CHRISTOFFERSEN INDEPENDENCE TEST
    # ============================================
    # H0: Breaches are independent (no clustering)
    # Test statistic follows chi-square distribution with 1 df
    
    # Count transitions: 00, 01, 10, 11
    n00 = n01 = n10 = n11 = 0
    for i in range(len(breaches) - 1):
        if breaches.iloc[i] == 0 and breaches.iloc[i+1] == 0:
            n00 += 1
        elif breaches.iloc[i] == 0 and breaches.iloc[i+1] == 1:
            n01 += 1
        elif breaches.iloc[i] == 1 and breaches.iloc[i+1] == 0:
            n10 += 1
        elif breaches.iloc[i] == 1 and breaches.iloc[i+1] == 1:
            n11 += 1
    
    # Calculate probabilities
    if n00 + n01 > 0:
        p01 = n01 / (n00 + n01)
    else:
        p01 = 0
        
    if n10 + n11 > 0:
        p11 = n11 / (n10 + n11)
    else:
        p11 = 0
    
    p = (n01 + n11) / (n00 + n01 + n10 + n11)
    
    # Calculate test statistic
    if p01 == 0 or p01 == 1 or p11 == 0 or p11 == 1 or p == 0 or p == 1:
        ind_stat = np.inf if p01 != p11 else 0
        ind_pvalue = 0.0 if p01 != p11 else 1.0
    else:
        ind_stat = -2 * (
            np.log((1-p)**(n00+n10) * p**(n01+n11)) -
            np.log((1-p01)**n00 * p01**n01 * (1-p11)**n10 * p11**n11)
        )
        ind_pvalue = 1 - stats.chi2.cdf(ind_stat, df=1)
    
    ind_critical = stats.chi2.ppf(confidence_level, df=1)
    ind_reject = ind_stat > ind_critical
    
    results['tests']['christoffersen_ind'] = {
        'statistic': ind_stat,
        'p_value': ind_pvalue,
        'critical_value': ind_critical,
        'reject_h0': ind_reject,
        'conclusion': 'REJECT - Breaches are clustered' if ind_reject else 'ACCEPT - Breaches are independent'
    }
    
    # ============================================
    # 3. CHRISTOFFERSEN COMBINED TEST (POF + IND)
    # ============================================
    # H0: Model is correct (both proportion and independence)
    # Test statistic follows chi-square distribution with 2 df
    
    combined_stat = pof_stat + ind_stat
    combined_pvalue = 1 - stats.chi2.cdf(combined_stat, df=2)
    combined_critical = stats.chi2.ppf(confidence_level, df=2)
    combined_reject = combined_stat > combined_critical
    
    results['tests']['christoffersen_combined'] = {
        'statistic': combined_stat,
        'p_value': combined_pvalue,
        'critical_value': combined_critical,
        'reject_h0': combined_reject,
        'conclusion': 'REJECT - Model is inadequate' if combined_reject else 'ACCEPT - Model is adequate'
    }
    
    # ============================================
    # OVERALL MODEL ASSESSMENT
    # ============================================
    all_pass = not (pof_reject or ind_reject or combined_reject)
    
    if all_pass:
        overall = "✓ MODEL ACCEPTABLE - All tests passed"
    elif pof_reject and ind_reject:
        overall = "✗ MODEL REJECTED - Both breach rate and independence issues"
    elif pof_reject:
        overall = "✗ MODEL REJECTED - Breach rate is significantly different from expected"
    elif ind_reject:
        overall = "⚠ MODEL QUESTIONABLE - Breaches show clustering (not independent)"
    else:
        overall = "✗ MODEL REJECTED - Combined test failed"
    
    results['overall_assessment'] = overall
    
    return results

def print_backtesting_results(results, model_name="VaR Model"):
    """
    Print formatted backtesting test results.
    """
    print("=" * 80)
    print(f"VaR BACKTESTING STATISTICAL TESTS - {model_name}")
    print("=" * 80)
    print(f"\nObservations: {results['n_observations']}")
    print(f"Breaches: {results['n_breaches']}")
    print(f"Breach Rate: {results['breach_rate']:.4f} ({results['breach_rate']*100:.2f}%)")
    print(f"Expected Rate: {results['expected_rate']:.4f} ({results['expected_rate']*100:.2f}%)")
    print(f"Difference: {(results['breach_rate'] - results['expected_rate'])*100:+.2f}%")
    
    print("\n" + "-" * 80)
    print("1. KUPIEC POF TEST (Proportion of Failures)")
    print("-" * 80)
    pof = results['tests']['kupiec_pof']
    print(f"H0: Breach rate = Expected rate ({results['expected_rate']*100:.0f}%)")
    print(f"Test Statistic: {pof['statistic']:.4f}")
    print(f"Critical Value ({results['expected_rate']*100:.2f}%): {pof['critical_value']:.4f}")
    print(f"P-value: {pof['p_value']:.4f}")
    print(f"Result: {pof['conclusion']}")
    
    print("\n" + "-" * 80)
    print("2. CHRISTOFFERSEN INDEPENDENCE TEST")
    print("-" * 80)
    ind = results['tests']['christoffersen_ind']
    print(f"H0: Breaches are independent (no clustering)")
    print(f"Test Statistic: {ind['statistic']:.4f}")
    print(f"Critical Value ({results['expected_rate']*100:.2f}%): {ind['critical_value']:.4f}")
    print(f"P-value: {ind['p_value']:.4f}")
    print(f"Result: {ind['conclusion']}")
    
    print("\n" + "-" * 80)
    print("3. CHRISTOFFERSEN COMBINED TEST")
    print("-" * 80)
    comb = results['tests']['christoffersen_combined']
    print(f"H0: Model is correct (both proportion and independence)")
    print(f"Test Statistic: {comb['statistic']:.4f}")
    print(f"Critical Value ({results['expected_rate']*100:.2f}%): {comb['critical_value']:.4f}")
    print(f"P-value: {comb['p_value']:.4f}")
    print(f"Result: {comb['conclusion']}")
    
    print("\n" + "=" * 80)
    print("OVERALL ASSESSMENT")
    print("=" * 80)
    print(f"{results['overall_assessment']}")
    print("=" * 80 + "\n")
