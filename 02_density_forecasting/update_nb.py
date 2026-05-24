import nbformat as nbf

notebook_path = r'c:\Users\fe_ma\Projects\quant-ai-lab\02_density_forecasting\notebooks\03_tf_example_02.ipynb'

# Read the notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# Create new cells
markdown_cell_1 = nbf.v4.new_markdown_cell('## 4. Synthetic Data Generation\nLet\'s test our models on synthetic data (AR(1)-GARCH(1,1) with Student-t innovations) to see how they perform when we know the true distribution has heavy tails and volatility clustering.')

code_cell_1 = nbf.v4.new_code_cell('''from src.data.synthetic import generate_student_t_garch
from src.features.features import create_features

# Generate 1500 days of synthetic data
df_synth_prices, df_synth_vix = generate_student_t_garch(n_days=1500, nu=5.0)

# Create features exactly as we do for real data
window_size = 22 * 6
df_synth_features = create_features(df_synth_prices, df_synth_vix, window_size, n_lags=2)
df_synth_features.dropna(inplace=True)
df_synth_features = df_synth_features.astype('float32')

df_synth_y = df_synth_features['returns']
df_synth_X = df_synth_features.drop(columns=['returns'])

df_synth_prices['returns'].plot(title='Synthetic Returns (AR1-GARCH1,1 Student-T)', figsize=(12, 4))
''')

markdown_cell_2 = nbf.v4.new_markdown_cell('## 5. Hyperparameter Tuning and Model Selection\nUsing `optuna` to search for the best model configuration. We evaluate models based on their probabilistic calibration (Block K-S Test on PIT) and their sharpness (CRPS).')

code_cell_2 = nbf.v4.new_code_cell('''from src.evaluation.tuning import hyperparameter_search

# Run a quick search on the synthetic data
print("Starting Optuna Hyperparameter Search...")
df_results = hyperparameter_search(
    df_X=df_synth_X,
    df_y=df_synth_y,
    n_trials=5,  # Keep it small for demonstration
    test_window=22 * 1,
    porcentage_train=0.8
)

display(df_results)
print("Best Configuration found:")
print(df_results.iloc[0])
''')

markdown_cell_3 = nbf.v4.new_markdown_cell('## 6. Baselines Comparison\nLet\'s compare our best neural network to simple historical baselines (Normal and Student-T).')

code_cell_3 = nbf.v4.new_code_cell('''from src.evaluation.tuning import evaluate_model_config

# Baseline: Historical Normal
config_normal = {'model_class': 'Historical_Normal'}
crps_n, ll_n, ks_fail_n = evaluate_model_config(df_synth_X, df_synth_y, config_normal, test_window=22, porcentage_train=0.8)

# Baseline: Historical Student-T
config_t = {'model_class': 'Historical_StudentT'}
crps_t, ll_t, ks_fail_t = evaluate_model_config(df_synth_X, df_synth_y, config_t, test_window=22, porcentage_train=0.8)

print(f"Historical Normal   -> CRPS: {crps_n:.4f}, KS Fail Rate: {ks_fail_n:.1%}")
print(f"Historical StudentT -> CRPS: {crps_t:.4f}, KS Fail Rate: {ks_fail_t:.1%}")
''')

# Append to notebook
nb.cells.extend([
    markdown_cell_1, code_cell_1, 
    markdown_cell_2, code_cell_2,
    markdown_cell_3, code_cell_3
])

# Write back
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
