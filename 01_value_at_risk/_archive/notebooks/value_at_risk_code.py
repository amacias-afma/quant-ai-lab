# --- CELL 3 ---
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

# --- CELL 4 ---
# Get the absolute path to the project root (one directory up from 'notebooks')
# Add it to sys.path if not already there

from value_at_risk.data.market import read_data
from value_at_risk.evaluation.backtest_value_at_risk import *

from value_at_risk.utils.clean_data import *
from value_at_risk.utils.plot_utils import *
from value_at_risk.utils.save_data import *

from value_at_risk.utils.var_functional_analysis import print_analysis_summary, plot_3d_surface_v2
from value_at_risk.utils.var_widgets import display_var_widgets

from value_at_risk.models.deep_var.features import *
from value_at_risk.models.deep_var.lstm_model import *
from value_at_risk.models.deep_var.parametric_model import *

os.makedirs('../images', exist_ok=True) # Ensure directory exists
os.makedirs('../results', exist_ok=True)

# --- CELL 6 ---
tickers = { "^GSPC": "S&P 500",
    "BTC-USD": "Bitcoin",
    "CLP=X": "USD/CLP (Chile Peso)",
    "SQM": "SQM (Lithium)",
    "HG=F": "Copper Futures",
    "TSLA": "Tesla",
    "NVDA": "NVIDIA",
    "CL=F": "Crude Oil",
    "TLT": "US Treasuries (20Y)",
    "VXX": "VIX Volatility"
    }

# --- CELL 7 ---

# Create additional parameter widgets
alpha_widget = widgets.Dropdown(
    options=[
        ('99% VaR (α=0.01)', 0.01),
        ('95% VaR (α=0.05)', 0.05),
        ('90% VaR (α=0.10)', 0.10)
    ],
    value=0.01,
    description='Risk Level:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='400px')
)

rolling_widget = widgets.Dropdown(
    options=[
        ('1 Month (22 days)', 22),
        ('3 Months (66 days)', 66),
        ('6 Months (132 days)', 132),
        ('1 Year (252 days)', 252)
    ],
    value=132,
    description='Rolling Window:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='400px')
)

epochs_widget = widgets.IntSlider(
    value=20000,
    min=500,
    max=20000,
    step=500,
    description='Epochs:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='400px'),
    continuous_update=False
)

lr_widget = widgets.FloatSlider(
    value=0.02,
    min=0.001,
    max=0.1,
    step=0.001,
    description='Learning Rate:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='400px'),
    continuous_update=False,
    readout_format='.3f'
)

# Display ticker and date widgets
ticker_widget, date_widget = display_var_widgets(
    default_ticker="BTC-USD",
    default_date="2026-01-31"
)

# Display additional parameter widgets
print("\n")
display(widgets.HTML("<h4>Model Parameters</h4>"))
display(alpha_widget)
display(rolling_widget)
display(epochs_widget)
display(lr_widget)

# --- CELL 9 ---
# === APPLY WIDGET SELECTIONS ===
# Extract values from widgets
TICKER = ticker_widget.value
DATE_REPORT = date_widget.value.strftime('%Y-%m-%d')
ALPHA = alpha_widget.value
ROLLING_WINDOW = rolling_widget.value
EPOCHS = epochs_widget.value
LEARNING_RATE = lr_widget.value

# Fixed parameters
MARKET_DATA_SOURCE = 'yfinance'
WEIGHT_PRIOR = 1.0

# Display current configuration
print("="*60)
print("CURRENT CONFIGURATION")
print("="*60)
print(f"Ticker:          {TICKER} ({tickers.get(TICKER, 'Unknown')})")
print(f"Report Date:     {DATE_REPORT}")
print(f"Risk Level:      {ALPHA*100}% VaR (α={ALPHA})")
print(f"Rolling Window:  {ROLLING_WINDOW} days")
print(f"Epochs:          {EPOCHS}")
print(f"Learning Rate:   {LEARNING_RATE}")
print(f"Prior Weight:    {WEIGHT_PRIOR}")
print("="*60)

# --- CELL 13 ---
# Load data
print(f"Loading data for {TICKER}...")
df = read_data(TICKER, market_data_source=MARKET_DATA_SOURCE, end_date=DATE_REPORT)
print(f"✓ Loaded {len(df)} observations")
print(f"  Date range: {df.index[0]} to {df.index[-1]}")

# --- CELL 15 ---
improved_price_plot(df, tickers[TICKER])

# Display sample
df.head()

# --- CELL 16 ---
# Basic statistics
print(f"Data points: {len(df)}")
print(f"Date range: {df.index[0]} to {df.index[-1]}")
print(f"\nPrice statistics:")
print(df['price'].describe())

# Calculate returns
returns = df['price'].pct_change().dropna()
print(f"\nReturn statistics:")
print(returns.describe())
print(f"Skewness: {returns.skew():.3f}")
print(f"Kurtosis: {returns.kurtosis():.3f}")

# --- CELL 19 ---
percentage_train = 0.7

train_int = int(len(df) * percentage_train)
split_date = df.index[train_int]
split_date

# --- CELL 20 ---
# Clean data
df, n_outliers = clean_data(df, z_score_threshold=10)
if n_outliers > 0:
    print(f"⚠️ Found and smoothed {n_outliers} outliers")
    improved_price_plot(df, tickers[TICKER])
else:
    print("✓ No outliers detected")

# --- CELL 22 ---
df['log_ret'] = np.log(df['price'] / df['price'].shift(1))
df.dropna(inplace=True)
improved_price_plot(df, tickers[TICKER], column='log_ret')

# --- CELL 25 ---
attempt_name = "1. Just Train the Model"
features = ['log_ret']
data = create_features(df, ALPHA, ROLLING_WINDOW, features=features)

# --- CELL 26 ---
# Train model
print(f"\n{'='*60}")
print(f"Training VaR Model - {TICKER}")
print(f"{'='*60}")
print(f"Risk Level (α): {ALPHA*100}%")
print(f"Training until: {split_date}")
print(f"Rolling window: {ROLLING_WINDOW} days")
print(f"{'='*60}\n")

split_type={'percentage': percentage_train}

model, history, train_data, test_data = train_model(
    data, 
    model_type='QuantileLSTM', 
    alpha=ALPHA, 
    epochs=EPOCHS, 
    lr=LEARNING_RATE, 
    rolling=ROLLING_WINDOW, 
    split_type=split_type,
    silent=True,
    regularization_pm=None
)

print(f"\n✓ Training complete!")
print(f"  Final train loss: {history['train_loss'][-1]:.5f}")
print(f"  Final test loss: {history['test_loss'][-1]:.5f}")

# --- CELL 28 ---
# Plot training history
plt.figure(figsize=(12, 5))
plt.plot(history['train_loss'], label='Training Loss', alpha=0.7)
plt.plot(history['test_loss'], label='Validation Loss', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Quantile Loss')
plt.title(f'Model Convergence for {attempt_name}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('../images/functional_loss.pdf', dpi=300, bbox_inches='tight')
plt.show()


# --- CELL 30 ---
df_results_04 = reestructure_testdata(test_data)
models_results = {attempt_name: df_results_04}
plot_var_results(df_results_04)

styled = df_results_04[['predicted']].describe().style
styled = styled.format("{:.0f}", subset=pd.IndexSlice['count', :])
styled = styled.format("{:.2%}", subset=pd.IndexSlice[['mean', 'std', 'min', '25%', '50%', '75%', 'max'], :])

styled

# --- CELL 33 ---
# --- 1. Solving the Scaling Issue ---
print("Applying Scaling Factor (x100)...")
attempt_name = '2. Back To Basics'
# We work with Percentages (e.g., -5.0 instead of -0.05) to help the optimizer
scale_factor = 100
df_scaled = df.copy()
df_scaled['log_ret'] = df_scaled['log_ret'] * scale_factor

# --- 2. Solving the Capacity Issue (Calculating Optimal Size) ---
train_samples = int(len(df_scaled) * 0.7)
max_params = train_samples // 10

print(f"Training Samples: {train_samples}")
print(f"Theoretical Max Parameters (to avoid overfitting): {max_params}")

# Let's calculate the size of our previous model vs. a Single Neuron
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Previous Architecture (The "Failure")
from value_at_risk.models.deep_var.lstm_model import QuantileLSTM 
prev_model = QuantileLSTM(input_size=1, hidden_size=64, num_layers=2)
print(f"Previous Model Size: {count_parameters(prev_model):,} parameters (❌ WAY TOO BIG)")

# New Proposed Architecture: Single Neuron (Linear Quantile Regression)
import torch.nn as nn
class SimpleQuantileNeuron(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1) # y = wx + b
    def forward(self, x):
        return self.linear(x)

tiny_model = SimpleQuantileNeuron(input_size=1)
print(f"New Model Size: {count_parameters(tiny_model)} parameters (✅ Safe Zone)")

# --- 3. Run the 'Right-Sized' Experiment ---
print("\nRetraining with Scaled Data + Tiny Model...")



# --- CELL 34 ---
features = ['log_ret']

data_scaled = create_features(df_scaled, ALPHA, ROLLING_WINDOW, features=features)

model_scaled, history_scaled, _, test_data_scaled = train_model(
    data_scaled, 
    model_type='SimpleQuantileNeuron', 
    alpha=ALPHA, 
    epochs=1000 * 10,
    lr=LEARNING_RATE, 
    rolling=ROLLING_WINDOW, 
    split_type=split_type,
    regularization_pm=None
)

print("Training Complete.")
# stop

# --- CELL 35 ---
df_results_05 = reestructure_testdata(test_data_scaled) / scale_factor
models_results[attempt_name] = df_results_05
plot_var_results(df_results_05)

styled = df_results_05[['predicted']].describe().style
styled = styled.format("{:.0f}", subset=pd.IndexSlice['count', :])
styled = styled.format("{:.2%}", subset=pd.IndexSlice[['mean', 'std', 'min', '25%', '50%', '75%', 'max'], :])

styled

# --- CELL 38 ---
# stop
attempt_name = '3. Feature Engineering'

scale_factor = 100
df_scaled = df.copy()
df_scaled['log_ret'] = df_scaled['log_ret'] * scale_factor

features = ['log_ret', 'std', 'log_ret^2']
data_featured = create_features(df_scaled, alpha=ALPHA, rolling=ROLLING_WINDOW, features=features)

# --- CELL 39 ---
# We use the SimpleQuantileNeuron now
model_featured, history_featured, _, test_data_featured = train_model(
    data_featured, 
    model_type='SimpleQuantileNeuron', 
    alpha=ALPHA, 
    epochs=EPOCHS, 
    lr=LEARNING_RATE, 
    rolling=ROLLING_WINDOW,
    split_type=split_type
)

print("Training Complete.")

# --- CELL 40 ---
df_results = reestructure_testdata(test_data_featured) / scale_factor
models_results[attempt_name] = df_results

plot_var_results(df_results)

styled = df_results[['predicted']].describe().style
styled = styled.format("{:.0f}", subset=pd.IndexSlice['count', :])
styled = styled.format("{:.2%}", subset=pd.IndexSlice[['mean', 'std', 'min', '25%', '50%', '75%', 'max'], :])

styled

# --- CELL 43 ---
attempt_name = '3.2. Walk-Forward Validation'

# --- CELL 44 ---
model_walk_forward, history_walk_forward, _, test_data_walk_forward = train_model(
    data_featured, 
    model_type='SimpleQuantileNeuron', 
    alpha=ALPHA, 
    epochs=EPOCHS, 
    lr=LEARNING_RATE, 
    rolling=ROLLING_WINDOW,
    split_type={'date': split_date}
)

print("Training Complete.")

# --- CELL 45 ---
df_results = reestructure_testdata(test_data_walk_forward) / scale_factor
models_results[attempt_name] = df_results

plot_var_results(df_results)

styled = df_results[['predicted']].describe().style
styled = styled.format("{:.0f}", subset=pd.IndexSlice['count', :])
styled = styled.format("{:.2%}", subset=pd.IndexSlice[['mean', 'std', 'min', '25%', '50%', '75%', 'max'], :])

styled

# --- CELL 47 ---
attempt_name = "4. Physics-Informed"
regularization_pm = {'weight': 0.005}

scale_factor = 100
df_scaled = df.copy()
df_scaled['log_ret'] = df_scaled['log_ret'] * scale_factor

regularization_pm['df'] = df_scaled

features = ['log_ret', 'std', 'log_ret^2']
data_physics = create_features(df_scaled, ALPHA, ROLLING_WINDOW, features=features)


# --- CELL 48 ---

# We use the SimpleQuantileNeuron now
model_physics, history_physics, _, test_data_physics = train_model(
    data_physics, 
    model_type='SimpleQuantileNeuron', 
    alpha=ALPHA, 
    epochs=EPOCHS, 
    lr=LEARNING_RATE, 
    rolling=ROLLING_WINDOW,
    split_type={'date': split_date},
    # pretrained_state_dict=model_walk_forward.state_dict(),
    regularization_pm=regularization_pm
)

print("Training Complete.")

# --- CELL 49 ---
df_results = reestructure_testdata(test_data_physics) / scale_factor
models_results[attempt_name] = df_results

plot_var_results(df_results)
styled = df_results[['predicted']].describe().style
styled = styled.format("{:.0f}", subset=pd.IndexSlice['count', :])
styled = styled.format("{:.2%}", subset=pd.IndexSlice[['mean', 'std', 'min', '25%', '50%', '75%', 'max'], :])

styled

# --- CELL 52 ---
df_results = pd.DataFrame()
for model, df_result in models_results.items():
    df_result_aux = df_result[['predicted']].copy()
    df_result_aux.columns = [model]
    df_results = pd.concat([df_results, df_result_aux], axis=1)

df_results = pd.concat([df_results, df_result[['realized']]], axis=1)
df_results.index = pd.to_datetime(df_results.index.get_level_values(0))

df_hist_var = df.copy()
column = '6. Historical VaR'
df_hist_var[column] = df_hist_var['log_ret'].rolling(252).quantile(0.01)
df_hist_var[column] = df_hist_var[column].shift(1)
df_hist_var.index = pd.to_datetime(df_hist_var.index)

df_results = pd.concat([df_results, df_hist_var[column]], axis=1)
df_results.dropna(inplace=True)

# --- CELL 53 ---
def calculate_metrics(df, alpha=0.01):
    """
    Generates a comparative table of VaR models.
    """
    metrics = []
    
    # Filter only model columns (excluding 'realized')
    model_cols = [c for c in df.columns if c not in ['realized', 'date']]
    
    for col in model_cols:
        # 1. Breach Rate (The Safety Check)
        # Breach = Realized < Predicted (e.g. -5% < -4%)
        breaches = df[df['realized'] < df[col]]
        n_breaches = len(breaches)
        n_total = len(df)
        breach_rate = n_breaches / n_total
        
        # 2. Kupiec Test (The Statistical Validation)
        # H0: Model matches target alpha. H1: It doesn't.
        # p-value < 0.05 means "Model is BROKEN"
        # p-value > 0.05 means "Model is PLAUSIBLE"
        try:
            p_val = stats.binomtest(n_breaches, n_total, alpha, alternative='two-sided').pvalue
        except AttributeError:
            # Fallback for older scipy versions
            p_val = 0.0 # Placeholder
            
        # 3. Capital Efficiency (Mean VaR)
        # We want this number to be small (closer to 0), but safe.
        avg_var = df[col].mean()
        
        # 4. Responsiveness (Std Dev)
        # Does the model react to vol, or is it a flat line?
        responsiveness = df[col].std()
        
        metrics.append({
            'Model': col,
            'Breach Rate (%)': f"{breach_rate*100:.2f}%",
            'Kupiec p-value': f"{p_val:.4f}",
            'Avg Capital Reserved': f"{avg_var*100:.2f}%",
            'Responsiveness (Std)': f"{responsiveness:.4f}",
            'Status': "✅ PASS" if p_val > 0.05 and abs(breach_rate - alpha) < 0.01 else "❌ FAIL"
        })
        
    return pd.DataFrame(metrics).set_index('Model')

# Load and Display
# results_df = pd.read_csv('results.csv', index_col=0, parse_dates=True)
summary_table = calculate_metrics(df_results)

print("\n=== 🏆 FINAL MODEL SHOWDOWN ===")
display(summary_table)

# --- CELL 56 ---
# df_results
metrics_to_export = {}

df_realized = df_results[['realized']]
df_results_aux = df_results.drop(columns='realized')
total_obs = len(df_realized)
for model in df_results_aux:
    print(model)
    if model[:3] != '3.2':
        df_realized[df_results_aux[[model]].values < df_realized.values]
        mean_var = df_results_aux[model].mean()
        breaches = len(df_realized[df_results_aux[[model]].values >= df_realized.values])
        # print(model, mean_var, breaches, total_obs)
        model_aux = model.replace(' ', '').replace('-', '')[2:]
        metrics_to_export[f'{model_aux}MeanVaR'] = f"{100 * mean_var:.2f}\\%"
        metrics_to_export[f'{model_aux}Breaches'] = breaches
        metrics_to_export[f'{model_aux}BreachRate'] = f"{100 * breaches / total_obs:.2f}\\%" 
        metrics_to_export[f'{model_aux}Zone'] = 'CORRECT'
        
        metrics_to_export[f'{model_aux}Status'] = summary_table.loc[model, 'Status']
        # paramBreaches
        # paramBreachRate

# 3. Write them to a .tex file
with open('../results/metrics.tex', 'w') as f:
    f.write("% Dynamically generated metrics from value_at_risk.ipynb\n")
    for command, value in metrics_to_export.items():
        f.write(f"\\newcommand{{\\{command}}}{{{value}}}\n")

print("Successfully exported LaTeX metrics to ../results/metrics.tex")

# --- CELL 57 ---
metrics_to_export

# --- CELL 58 ---
# 2. Define the exact metrics you calculated in your notebook
# (Replace the right-hand side with your actual python variables from the notebook)
metrics_to_export = {
    # Parametric Baseline
    'paramBreaches': 3,                 # e.g., your parametric_breach_count variable
    'paramBreachRate': '1.20\\%',       # e.g., f"{(parametric_breach_count/250)*100:.2f}\\%"
    'paramMeanVar': '4.50\\%',          # e.g., f"{parametric_mean_var:.2f}\\%"
    
    # Naive LSTM
    'naiveBreaches': 14,                # e.g., your naive_breach_count variable
    'naiveBreachRate': '5.60\\%', 
    'naiveZone': 'Red Zone',            # e.g., "Red Zone" if breaches >= 10
    'naiveMeanVar': '2.10\\%',
    
    # Anchored LSTM (Proposed)
    'anchoredBreaches': 4,              # e.g., your anchored_breach_count variable
    'anchoredBreachRate': '1.60\\%',
    'anchoredMeanVar': '2.80\\%',
    
    # The Hook / ROI
    'capEfficiency': '38.5\\%'          # e.g., f"{((param_mean_var - anchored_mean_var) / param_mean_var)*100:.1f}\\%"
}

# 3. Write them to a .tex file
with open('../results/metrics.tex', 'w') as f:
    f.write("% Dynamically generated metrics from value_at_risk.ipynb\n")
    for command, value in metrics_to_export.items():
        f.write(f"\\newcommand{{\\{command}}}{{{value}}}\n")

print("Successfully exported LaTeX metrics to ../results/metrics.tex")

# --- CELL 59 ---
model_1 = '1. Just Train the Model'
model_4 = '4. Physics-Informed'
model_6 = '6. Historical VaR'

plot_var_results(df_results, models={model_1, model_4, model_6}, file_name='compare_results.pdf')
# compare_two_models(df_results, model_1, model_2)
# save_comparison_plot(df_results, model_1, model_2)
# save_table_image(summary_table)

# --- CELL 66 ---
import json
import os

results_path = '../outputs/var_results.json'
os.makedirs(os.path.dirname(results_path), exist_ok=True)

try:
    with open(results_path, 'r', encoding='utf-8') as f:
        all_results = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    all_results = {}

# Convert summary table to dictionary
dict_results = summary_table.to_dict(orient='index')
all_results[TICKER] = dict_results

with open(results_path, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=4)

print(f"Saved results for {TICKER} to {results_path}")

# --- CELL 67 ---
import json
import os
import numpy as np

ts_path = '../outputs/var_timeseries.json'
os.makedirs(os.path.dirname(ts_path), exist_ok=True)

try:
    with open(ts_path, 'r', encoding='utf-8') as f:
        all_ts = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    all_ts = {}

# Convert df_results to dictionary
df_ts = df_results.copy()
df_ts.index = df_ts.index.astype(str)

# Replace NaN with None for JSON serialization
df_ts = df_ts.replace({np.nan: None})

all_ts[TICKER] = df_ts.to_dict(orient='list')
all_ts[TICKER]['date'] = df_ts.index.tolist()

with open(ts_path, 'w', encoding='utf-8') as f:
    json.dump(all_ts, f, indent=4)

print(f"Saved timeseries for {TICKER} to {ts_path}")