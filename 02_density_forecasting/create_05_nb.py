import nbformat as nbf

notebook_path = r'c:\Users\fe_ma\Projects\quant-ai-lab\02_density_forecasting\notebooks\05_vix_baseline_vs_ml.ipynb'
nb = nbf.v4.new_notebook()

# Cell 1: Intro
md_1 = nbf.v4.new_markdown_cell('''# Chapter 5: Advanced Parametric vs The ML Floor (Fast Multi-Ticker Showdown)

This notebook implements the hyper-optimized pipeline to compare the `VIX-Scaled Student-T` baseline against the `WideAndDeep` Neural Network over a rigorous expanding window walk-forward backtest.

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
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from src.evaluation.metrics import block_ks_test
from src.evaluation.metrics import evaluate_forecasts
from src.models.neural_networks import generate_montecarlo

# Import our optimized pipeline functions
from src.evaluation.fast_pipeline import (
    read_data_features,
    extract_array,
    extract_df_nn,
    run_baseline,
    backtesting_baseline,
    initializate_nn_model,
    run_neural_networks,
    backtesting_neural_networks
)
''')

# Cell 3: Config
md_2 = nbf.v4.new_markdown_cell('## 1. Configuration & Multi-Ticker Loop')
code_2 = nbf.v4.new_code_cell('''tickers = ['ARKK', 'USO', 'USDCLP=X', 'BTC-USD', 'SQM-B.SN']
start_date = date(2015, 12, 31)
end_date = date(2025, 12, 31)

columns_linear = ['ret_1_mean', 'ret_1_std', 'ret_1_kurt', 'ret_1_skew', 'ret_1_mean_st', 'ret_1_std_st', 'vix_ratio', 'VIX']
columns_deep = ['vix_ratio']

test_window = 22
porcentage_train = 0.70

initial_epochs = 501
model_class = 'WideAndDeep'
lr = 0.015

table_results = []
timeseries_results = {}
''')

# Cell 4: The Master Loop
code_3 = nbf.v4.new_code_cell('''for ticker in tickers:
    print(f"\\n{'='*50}")
    print(f"STARTING FAST SHOWDOWN FOR {ticker}")
    print(f"{'='*50}")
    
    # Setup Data
    df_features = read_data_features(ticker, start_date, end_date)
    
    dates_test = []
    pred_mu_result = {'baseline': [], 'neural-network': []}
    pred_sigma_result = {'baseline': [], 'neural-network': []}
    pred_nu_result = {'baseline': [], 'neural-network': []}
    pit_values_result = {'baseline': [], 'neural-network': []}
    
    n_train = int(porcentage_train * len(df_features))
    current_epochs = initial_epochs
    cold_start = True
    last_optimal_beta = 0.5
    
    # Initialize NN once per ticker
    train_step, model = initializate_nn_model(columns_linear, model_class, lr, columns_deep=columns_deep)
    
    print(f"Expanding Window Walk-Forward (Start Train Size: {n_train})...")
    
    while n_train < len(df_features):
        df_features_train = df_features.iloc[:n_train]
        df_X_linear_train, df_X_deep_train, df_y_train = extract_df_nn(df_features_train, columns_linear, columns_deep)

        df_features_test = df_features.iloc[n_train:n_train + test_window]
        df_X_linear_test, df_X_deep_test, df_y_test = extract_df_nn(df_features_test, columns_linear, columns_deep)
        
        # We need this to check if there are any days to test
        if len(df_features_test) == 0:
            break

        dates_test.append(df_features_test.index)

        # Baseline
        data_train = extract_array(df_features_train, df_y_train)
        optimal_beta = run_baseline(data_train, last_optimal_beta, cold_start)
        last_optimal_beta = optimal_beta
        
        data_test = extract_array(df_features_test, df_y_test)
        baseline_results = backtesting_baseline(data_test, optimal_beta)

        pred_mu_result['baseline'].append(baseline_results['mu'])
        pred_sigma_result['baseline'].append(baseline_results['sigma'])
        pred_nu_result['baseline'].append(baseline_results['nu'])
        pit_values_result['baseline'].append(baseline_results['pit_values'])

        # Neural Network
        run_neural_networks(df_X_linear_train, df_X_deep_train, df_y_train, current_epochs, train_step, verbose=False)
        if cold_start:
            current_epochs = current_epochs // 2
            cold_start = False

        neural_networks_results = backtesting_neural_networks(
            tf.convert_to_tensor(df_X_linear_test.to_numpy(), dtype=tf.float32), 
            tf.convert_to_tensor(df_X_deep_test.to_numpy(), dtype=tf.float32) if df_X_deep_test is not None else None, 
            tf.convert_to_tensor(df_y_test.to_numpy(), dtype=tf.float32), 
            model
        )

        pred_mu_result['neural-network'].append(neural_networks_results['mu'])
        pred_sigma_result['neural-network'].append(neural_networks_results['sigma'])
        pred_nu_result['neural-network'].append(neural_networks_results['nu'])
        pit_values_result['neural-network'].append(neural_networks_results['pit_values'])

        n_train += test_window

    # Combine Results for this Ticker
    dates_test_total = pd.Index(np.concatenate(dates_test))
    actual_returns = df_features['returns'].loc[dates_test_total]
    
    # Helper to compute metrics
    def calculate_metrics(model_name):
        mu_total = np.concatenate(pred_mu_result[model_name])
        sigma_total = np.concatenate(pred_sigma_result[model_name])
        nu_total = np.concatenate(pred_nu_result[model_name])
        pit_total = np.concatenate(pit_values_result[model_name])
        
        # Monte carlo for CRPS
        ensembles = generate_montecarlo(mu_total, sigma_total, nu_total, n_samples=1000)
        df_eval = evaluate_forecasts(actual_returns, ensembles)
        
        crps = df_eval['CRPS'].mean()
        
        # Block KS
        df_ks = block_ks_test(pit_total, block_size=60, alpha=0.05)
        ks_fail = (df_ks['Status'] == '❌ FAILED').mean() if len(df_ks) > 0 else 1.0
        
        return crps, ks_fail
        
    crps_base, ks_base = calculate_metrics('baseline')
    crps_nn, ks_nn = calculate_metrics('neural-network')
    
    table_results.append({
        'Ticker': ticker,
        'CRPS_Vix_StudentT': crps_base,
        'CRPS_WideDeep_ML': crps_nn,
        'CRPS_Winner': 'ML' if crps_nn < crps_base else 'VIX-StudentT',
        'KS_Fail_Vix_StudentT': ks_base,
        'KS_Fail_WideDeep_ML': ks_nn,
        'KS_Winner': 'ML' if ks_nn < ks_base else ('Tie' if ks_nn == ks_base else 'VIX-StudentT')
    })
    
    print(f"Metrics -> Baseline CRPS: {crps_base:.4f} | ML CRPS: {crps_nn:.4f}")

    # --- BLOCK PLOT (2x2) ---
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Density Forecast Comparison: {ticker}", fontsize=16)

    mu_base = np.concatenate(pred_mu_result['baseline'])
    mu_nn = np.concatenate(pred_mu_result['neural-network'])
    axes[0, 0].plot(dates_test_total, actual_returns, label='Actual Returns', alpha=0.5, color='gray')
    axes[0, 0].plot(dates_test_total, mu_base, label='Baseline Mu', alpha=0.8, color='blue')
    axes[0, 0].plot(dates_test_total, mu_nn, label='ML Mu', alpha=0.8, color='orange')
    axes[0, 0].set_title('Returns vs Predicted Mu')
    axes[0, 0].legend()

    sigma_base = np.concatenate(pred_sigma_result['baseline'])
    sigma_nn = np.concatenate(pred_sigma_result['neural-network'])
    axes[0, 1].plot(dates_test_total, sigma_base, label='Baseline Sigma', alpha=0.8, color='blue')
    axes[0, 1].plot(dates_test_total, sigma_nn, label='ML Sigma', alpha=0.8, color='orange')
    axes[0, 1].set_title('Predicted Sigma (Volatility)')
    axes[0, 1].legend()

    nu_base = np.concatenate(pred_nu_result['baseline'])
    nu_nn = np.concatenate(pred_nu_result['neural-network'])
    axes[1, 0].plot(dates_test_total, nu_base, label='Baseline Nu', alpha=0.8, color='blue')
    axes[1, 0].plot(dates_test_total, nu_nn, label='ML Nu', alpha=0.8, color='orange')
    axes[1, 0].set_title('Predicted Nu (Degrees of Freedom)')
    axes[1, 0].legend()

    pit_base = np.concatenate(pit_values_result['baseline'])
    pit_nn = np.concatenate(pit_values_result['neural-network'])
    sns.histplot(pit_base, bins=20, color='blue', alpha=0.4, label='Baseline', ax=axes[1, 1], stat='density')
    sns.histplot(pit_nn, bins=20, color='orange', alpha=0.4, label='ML', ax=axes[1, 1], stat='density')
    axes[1, 1].axhline(1.0, color='red', linestyle='--', label='Ideal Uniform')
    axes[1, 1].set_title('PIT Distribution')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()
    
    timeseries_results[ticker] = {
        'dates': [d.strftime('%Y-%m-%d') for d in dates_test_total],
        'actual_returns': actual_returns.tolist(),
        'mu_base': mu_base.tolist(),
        'sigma_base': sigma_base.tolist(),
        'nu_base': nu_base.tolist(),
        'mu_nn': mu_nn.tolist(),
        'sigma_nn': sigma_nn.tolist(),
        'nu_nn': nu_nn.tolist()
    }
''')

# Cell 5: Results Table
md_4 = nbf.v4.new_markdown_cell('## 2. Final Comparison Table')
code_4 = nbf.v4.new_code_cell('''df_results = pd.DataFrame(table_results)
df_results.set_index('Ticker', inplace=True)

# Format for pretty display
df_results['KS_Fail_Vix_StudentT'] = df_results['KS_Fail_Vix_StudentT'].map(lambda x: f"{x:.1%}")
df_results['KS_Fail_WideDeep_ML'] = df_results['KS_Fail_WideDeep_ML'].map(lambda x: f"{x:.1%}")
df_results['CRPS_Vix_StudentT'] = df_results['CRPS_Vix_StudentT'].map(lambda x: f"{x:.4f}")
df_results['CRPS_WideDeep_ML'] = df_results['CRPS_WideDeep_ML'].map(lambda x: f"{x:.4f}")

display(df_results)
''')

# Cell 6: Save Results
code_5 = nbf.v4.new_code_cell('''import json
import os
output_dir = '../outputs'
os.makedirs(output_dir, exist_ok=True)
with open(f'{output_dir}/df_timeseries.json', 'w') as f:
    json.dump(timeseries_results, f)
df_results.to_csv(f'{output_dir}/showdown_results.csv')
df_results.to_json(f'{output_dir}/showdown_results.json', orient='index')
print(f'Results saved to {output_dir}')
''')

nb.cells = [md_1, code_1, md_2, code_2, code_3, md_4, code_4, code_5]

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook generated successfully at {notebook_path}")
