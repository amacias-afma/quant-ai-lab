import nbformat as nbf

notebook_path = r'c:\Users\fe_ma\Projects\quant-ai-lab\02_density_forecasting\notebooks\05_vix_baseline_vs_ml.ipynb'

nb = nbf.v4.new_notebook()

# Cell 1: Intro
md_1 = nbf.v4.new_markdown_cell('''# Chapter 5: Advanced Parametric vs The ML Floor

We now compare the advanced VIX-Scaled Student-T (which learns VIX elasticity via MLE) against the Wide & Deep Neural Network. 
Both models are tested on the ARKK dataset using a rigorous walk-forward backtest.

We measure:
1. **CRPS (Sharpness)**: Lower is better.
2. **Block K-S Failure Rate (Calibration)**: What percentage of 60-day windows fail the K-S test? (Lower is better).''')

# Cell 2: Imports
code_1 = nbf.v4.new_code_cell('''import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('..')

from datetime import date
import pandas as pd
import numpy as np

from src.data.data_loader import fetch_asset_data
from src.features.features import create_features
from src.evaluation.tuning import evaluate_model_config
''')

# Cell 3: Data and Feature Engineering
md_2 = nbf.v4.new_markdown_cell('## 1. Data Fetching & Feature Engineering')
code_2 = nbf.v4.new_code_cell('''ticker = 'ARKK'
start_date = date(2015, 12, 31)
end_date = date(2025, 12, 31)

print(f"Fetching {ticker} data...")
df_asset = fetch_asset_data(ticker=ticker, start=start_date, end=end_date, features=[])
print("Fetching VIX data...")
df_vix_raw = fetch_asset_data(ticker='^VIX', start=start_date, end=end_date, features=[])

# Adjust VIX to be daily standard deviation in percentage points
df_vix = (df_vix_raw[['prices']] / np.sqrt(252)) * 100
df_vix.columns = ['VIX']

# Apply custom feature engineering
window_size_long = 22 * 6
df_features = create_features(df_asset, df_vix, window_size_long, n_lags=3)

window_size_short = int(22 / 2)
df_features_st = create_features(df_asset, df_vix, window_size_short, n_lags=1)
df_features_st.dropna(inplace=True)
df_features_st = df_features_st.astype('float32')
df_features_st = df_features_st[['ret_1_mean', 'ret_1_std', 'ret_1_kurt', 'ret_1_skew']]
df_features_st.columns = ['ret_1_mean_st', 'ret_1_std_st', 'ret_1_kurt_st', 'ret_1_skew_st']

df_features = pd.concat([df_features, df_features_st], axis=1)
df_features.dropna(inplace=True)
df_features = df_features.astype('float32')

columns_linear = ['ret_1_mean', 'ret_1_std', 'ret_1_kurt', 'ret_1_skew', 'ret_1_mean_st', 'ret_1_std_st', 'VIX']
columns_deep = ['VIX', 'ret_1', 'ret_2', 'ret_3']

df_X_linear = df_features[columns_linear]
df_X_deep = df_features[columns_deep]
df_y = df_features['returns']

print(f"Total trading days ready for backtest: {len(df_y)}")
''')

# Cell 4: Parametric Baselines
md_3 = nbf.v4.new_markdown_cell('## 2. Advanced Parametric: VIX-Scaled Student-T')
code_3 = nbf.v4.new_code_cell('''test_window = 22
porcentage_train = 0.60

print("--- EVALUATING VIX-SCALED STUDENT-T BASELINE ---")
crps_vix, ll_vix, ks_fail_vix = evaluate_model_config(
    df_X_linear=df_X_linear, 
    df_y=df_y, 
    config={'model_class': 'VIX_Scaled_StudentT'},
    df_X_deep=df_X_deep, # Pass but ignored by baseline
    test_window=test_window, 
    porcentage_train=porcentage_train
)
print(f"VIX-Scaled Student-T -> CRPS: {crps_vix:.4f} | Block KS Fail Rate: {ks_fail_vix:.1%}\\n")
''')

# Cell 5: ML Models
md_4 = nbf.v4.new_markdown_cell('## 3. The Machine Learning Floor')
code_4 = nbf.v4.new_code_cell('''print("--- EVALUATING WIDE & DEEP NEURAL NETWORK ---")
config_ml = {
    'model_class': 'WideAndDeep',
    'epochs': 150,
    'lr': 0.005
}

crps_ml, ll_ml, ks_fail_ml = evaluate_model_config(
    df_X_linear=df_X_linear, 
    df_y=df_y, 
    config=config_ml,
    df_X_deep=df_X_deep,
    test_window=test_window, 
    porcentage_train=porcentage_train
)

print("\\n\\n=========================================")
print("          FINAL SHOWDOWN RESULTS         ")
print("=========================================")
print(f"VIX-Scaled Student-T | CRPS: {crps_vix:.4f} | KS Fail Rate: {ks_fail_vix:.1%}")
print(f"Wide & Deep ML       | CRPS: {crps_ml:.4f} | KS Fail Rate: {ks_fail_ml:.1%}")
print("=========================================")
''')

nb.cells = [md_1, code_1, md_2, code_2, md_3, code_3, md_4, code_4]

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook generated successfully at {notebook_path}")
