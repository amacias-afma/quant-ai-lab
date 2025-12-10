import pandas as pd
from statsmodels.tsa.stattools import adfuller

def adf_test(series, title=''):
    """
    Pass in a time-series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    # Drop NaN values usually required for adfuller
    result = adfuller(series.dropna(), autolag='AIC') 
    
    labels = ['ADF Test Statistic','p-value','# Lags Used','Number of Observations Used']
    out = pd.Series(result[0:4], index=labels)
    
    for key,val in result[4].items():
        out[f'Critical Value ({key})'] = val
        
    print(out)
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is STATIONARY")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is NON-STATIONARY")
    print("\n-------------------------------------------\n")
