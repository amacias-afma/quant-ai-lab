import numpy as np
from scipy import stats
import pandas as pd

def clean_data(df, z_score_threshold=4):
    """
    1. Fills NaNs with the last known value (Forward Fill).
    2. Detects statistical outliers in price changes and replaces them.
    """
    df_clean = df.copy()
    
    # 1. Handle Missing Data
    if df_clean.isnull().sum().sum() > 0:
        print(f"Found {df_clean.isnull().sum().sum()} NaNs. Forward filling...")
        df_clean.ffill(inplace=True)
    
    # 2. Outlier Detection (Simple Z-Score on diffs)
    # We look for unrealistic jumps in price
    price_diff = df_clean['price'].diff()
    z_scores = np.abs(stats.zscore(price_diff.dropna()))
    
    # Align Z-scores to original index
    outliers = z_scores > z_score_threshold
    
    if np.sum(outliers) > 0:
        print(f"⚠️ Found {np.sum(outliers)} outliers. Smoothing...")
        # Replace outlier prices with the previous day's price
        # (A simplified approach; rigorous cleaning might use interpolation)
        # Note: We perform this on indices where Z-score is high
        outlier_indices = price_diff.dropna()[outliers].index
        for idx in outlier_indices:
            df_clean.loc[idx, 'price'] = df_clean.loc[:idx].iloc[-2]['price'] # Set to prev value
            
    return df_clean, np.sum(outliers)

def reestructure_testdata(test_data):
    X_test, y_test, preds_test, dates_test = test_data
    df_results = pd.DataFrame()
    for i in range(len(dates_test)):
        # print(i, len(dates_test[i]))
        if len(dates_test[i]) > 0:
            df_results_aux = pd.DataFrame([], index=[dates_test[i]])
            # Squeeze arrays to handle varying dimensions (e.g., (731, 1, 1) -> (731, 1))
            y_squeezed = np.squeeze(y_test[i])
            preds_squeezed = np.squeeze(preds_test[i])
            df_results_aux.loc[dates_test[i], 'realized'] = pd.DataFrame(y_squeezed).values
            df_results_aux.loc[dates_test[i], 'predicted'] = pd.DataFrame(preds_squeezed).values
            df_results = pd.concat([df_results, df_results_aux])
    return df_results