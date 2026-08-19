import nbformat as nbf

notebook_path = r'c:\Users\fe_ma\Projects\quant-ai-lab\02_density_forecasting\notebooks\04_ml_vs_parametric.ipynb'

nb = nbf.v4.new_notebook()

# Cell 1: Intro
md_1 = nbf.v4.new_markdown_cell('''# Chapter 4: The Parametric Ceiling vs The ML Floor

In Chapter 2, we saw that classical parametric models (like the rolling Student-t or Historical Normal) look good on average, but completely break down during market regime shifts (like the 2020 COVID crash or 2022 bear market). They suffer from **Unconditional Calibration**.

In this notebook, we pit those exact baselines against our new **Wide & Deep Student-t Neural Network**. We evaluate them on the `ARKK` dataset using our rigorous walk-forward backtest framework. 

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
from src.models.tf_neural_networks import train_expanding_window_model
''')

# Cell 3: Data
md_2 = nbf.v4.new_markdown_cell('## 1. Data Fetching & Fast Feature Engineering')
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

# Create features using the ultra-fast empirical moments!
window_size = 22 * 6 # 6 months
df_features = create_features(df_asset, df_vix, window_size, n_lags=2)
df_features.dropna(inplace=True)
df_features = df_features.astype('float32')

df_y = df_features['returns']
df_X = df_features.drop(columns=['returns'])

print(f"Total trading days ready for backtest: {len(df_X)}")
''')

# Cell 4: Parametric Baselines
md_3 = nbf.v4.new_markdown_cell('## 2. The Parametric Ceiling')
code_3 = nbf.v4.new_code_cell('''# We use a 22-day walk-forward step. Initial training/warmup is 60% of the dataset.
test_window = 22
porcentage_train = 0.60

print("--- EVALUATING HISTORICAL NORMAL BASELINE ---")
crps_norm, ll_norm, ks_fail_norm = evaluate_model_config(
    df_X, df_y, 
    config={'model_class': 'Historical_Normal'},
    test_window=test_window, 
    porcentage_train=porcentage_train
)
print(f"Historical Normal -> CRPS: {crps_norm:.4f} | Block KS Fail Rate: {ks_fail_norm:.1%}\\n")

print("--- EVALUATING HISTORICAL STUDENT-T BASELINE ---")
crps_t, ll_t, ks_fail_t = evaluate_model_config(
    df_X, df_y, 
    config={'model_class': 'Historical_StudentT'},
    test_window=test_window, 
    porcentage_train=porcentage_train
)
print(f"Historical Student-T -> CRPS: {crps_t:.4f} | Block KS Fail Rate: {ks_fail_t:.1%}\\n")
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
    df_X, df_y, 
    config=config_ml,
    test_window=test_window, 
    porcentage_train=porcentage_train
)

print("\\n\\n=========================================")
print("          FINAL SHOWDOWN RESULTS         ")
print("=========================================")
print(f"Historical Normal    | CRPS: {crps_norm:.4f} | KS Fail Rate: {ks_fail_norm:.1%}")
print(f"Historical Student-T | CRPS: {crps_t:.4f} | KS Fail Rate: {ks_fail_t:.1%}")
print(f"Wide & Deep ML       | CRPS: {crps_ml:.4f} | KS Fail Rate: {ks_fail_ml:.1%}")
print("=========================================")
''')

nb.cells = [md_1, code_1, md_2, code_2, md_3, code_3, md_4, code_4]

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook generated successfully at {notebook_path}")
