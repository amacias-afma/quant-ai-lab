import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell('# Value at Risk: Modular Framework\nThis notebook builds a modular VaR evaluation pipeline from scratch, allowing testing of different parameters (like rolling windows and alpha levels).'))

cells.append(nbf.v4.new_code_cell('''import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

sns.set_theme(style="whitegrid")
import warnings
warnings.filterwarnings("ignore")'''))

cells.append(nbf.v4.new_markdown_cell('## 1. Fetching Data'))

cells.append(nbf.v4.new_code_cell('''def fetch_data(ticker, start_date="2010-01-01", end_date=None):
    """Fetches historical price data from Yahoo Finance."""
    print(f"Fetching data for {ticker}...")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    # Handle potentially multi-level columns if multiple tickers are passed or newer yfinance behavior
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.droplevel(1)
        except:
            pass
            
    if 'Adj Close' in df.columns:
        prices = df['Adj Close']
    else:
        prices = df['Close']
    return pd.DataFrame({'price': prices})

ticker = "^GSPC" # Example: S&P 500
df_raw = fetch_data(ticker)
df_raw.head()'''))

cells.append(nbf.v4.new_markdown_cell('## 2. Preprocessing & Outlier Removal'))

cells.append(nbf.v4.new_code_cell('''def preprocess_data(df, column='price', z_thresh=3.0):
    """
    Cleans the data by removing extreme outliers based on rolling z-scores 
    of returns to avoid look-ahead bias.
    """
    df_clean = df.copy()
    df_clean.dropna(inplace=True)
    
    # Calculate initial daily returns for outlier detection
    temp_returns = df_clean[column].pct_change().dropna()
    
    # Calculate expanding window mean and std for z-score (no look-ahead)
    expanding_mean = temp_returns.expanding(min_periods=30).mean()
    expanding_std = temp_returns.expanding(min_periods=30).std()
    
    # Calculate Z-scores
    z_scores = np.abs((temp_returns - expanding_mean) / expanding_std)
    
    # Identify outliers
    outliers = z_scores > z_thresh
    
    # Remove outliers from original dataframe
    dates_to_keep = outliers[~outliers].index
    df_clean = df_clean.loc[df_clean.index.intersection(dates_to_keep)]
    
    print(f"Removed {outliers.sum()} extreme outliers.")
    return df_clean

df_clean = preprocess_data(df_raw)'''))

cells.append(nbf.v4.new_markdown_cell('## 3. Calculating Returns'))

cells.append(nbf.v4.new_code_cell('''def calculate_returns(df, column='price'):
    """Calculates logarithmic returns."""
    df_ret = df.copy()
    df_ret['log_return'] = np.log(df_ret[column] / df_ret[column].shift(1))
    df_ret.dropna(inplace=True)
    return df_ret

df_returns = calculate_returns(df_clean)
df_returns['log_return'].plot(title=f'{ticker} Log Returns', figsize=(12, 4))
plt.show()'''))

cells.append(nbf.v4.new_markdown_cell('## 4. Modular VaR Models\nHere we define our VaR models as functions where we can easily inject parameters like the rolling window and alpha.'))

cells.append(nbf.v4.new_code_cell('''def historical_var_model(returns, window=252, alpha=0.01):
    """
    Simple Modular VaR Model using Historical Simulation.
    Calculates the VaR for the NEXT day using the past 'window' days.
    """
    # Calculate the rolling quantile. 
    # The VaR calculated today applies to tomorrow, so we shift by 1.
    var_prediction = returns.rolling(window=window).quantile(alpha).shift(1)
    return var_prediction

def parametric_var_model(returns, window=252, alpha=0.01):
    """
    Simple Modular VaR Model using Parametric (Normal) assumption.
    """
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()
    z_score = norm.ppf(alpha)
    
    var_prediction = (rolling_mean + z_score * rolling_std).shift(1)
    return var_prediction'''))

cells.append(nbf.v4.new_markdown_cell('## 5. Model Evaluation vs Actuals'))

cells.append(nbf.v4.new_code_cell('''def evaluate_var(actual_returns, var_predictions):
    """
    Evaluates the VaR predictions against actual returns.
    Creates a dataframe with the results and identifies breaches.
    """
    df_eval = pd.DataFrame({
        'Realized_Return': actual_returns,
        'Predicted_VaR': var_predictions
    }).dropna()
    
    # A breach occurs if the actual return is worse (more negative) than the predicted VaR
    df_eval['Breach'] = (df_eval['Realized_Return'] < df_eval['Predicted_VaR']).astype(int)
    
    return df_eval

# Example usage for one model
window_size = 252
alpha_level = 0.01

var_hist = historical_var_model(df_returns['log_return'], window=window_size, alpha=alpha_level)
df_results = evaluate_var(df_returns['log_return'], var_hist)
df_results.tail()'''))

cells.append(nbf.v4.new_markdown_cell('## 6. Rolling Statistic of Results\nWe calculate the percentage of breaches over a rolling window to see how performance fluctuates over time.'))

cells.append(nbf.v4.new_code_cell('''def calculate_rolling_breach_rate(df_eval, rolling_window=252):
    """
    Calculates the rolling percentage of breaches over a given window.
    """
    df_eval = df_eval.copy()
    # Moving average of the Breach column gives the rolling breach rate
    df_eval['Rolling_Breach_Rate'] = df_eval['Breach'].rolling(window=rolling_window).mean()
    return df_eval

df_results_stats = calculate_rolling_breach_rate(df_results, rolling_window=252)

# Plotting the rolling breach rate
plt.figure(figsize=(12, 4))
plt.plot(df_results_stats.index, df_results_stats['Rolling_Breach_Rate'] * 100, label='Rolling Breach Rate (252d)')
plt.axhline(y=alpha_level * 100, color='r', linestyle='--', label=f'Theoretical Target ({alpha_level*100}%)')
plt.title('Rolling Breach Rate vs Theoretical Target')
plt.ylabel('Breach Rate (%)')
plt.legend()
plt.show()'''))

cells.append(nbf.v4.new_markdown_cell('## 7. Model and Parameter Comparison\nHere we run a battery of tests over multiple models and parameters to compare them.'))

cells.append(nbf.v4.new_code_cell('''def run_model_comparison(returns_series, models_to_test, test_alpha=0.01):
    """
    Runs multiple models and parameter sets, returning a summary table and a unified dataframe of rolling stats.
    """
    from scipy.stats import binomtest
    summary_stats = []
    df_rolling_rates = pd.DataFrame(index=returns_series.index)
    
    for config in models_to_test:
        name = config['name']
        func = config['func']
        kwargs = config['kwargs']
        
        # 1. Run model
        var_preds = func(returns_series, **kwargs)
        
        # 2. Evaluate
        df_eval = evaluate_var(returns_series, var_preds)
        
        # 3. Overall Stats
        total_days = len(df_eval)
        total_breaches = df_eval['Breach'].sum()
        overall_breach_rate = total_breaches / total_days if total_days > 0 else 0
        
        # Kupiec POF Test: Evaluates how likely the observed breaches are under the theoretical alpha
        try:
            kupiec_p = binomtest(total_breaches, total_days, kwargs.get('alpha', test_alpha), alternative='two-sided').pvalue
        except AttributeError:
            kupiec_p = 0.0
            
        summary_stats.append({
            'Model Name': name,
            'Window': kwargs.get('window', 'N/A'),
            'Alpha': kwargs.get('alpha', test_alpha),
            'Total Test Days': total_days,
            'Total Breaches': total_breaches,
            'Overall Breach Rate (%)': round(overall_breach_rate * 100, 3),
            'Kupiec p-value': round(kupiec_p, 4)
        })
        
        # 4. Rolling Stats
        df_eval_roll = calculate_rolling_breach_rate(df_eval, rolling_window=252)
        df_rolling_rates[name] = df_eval_roll['Rolling_Breach_Rate'] * 100
        
    df_summary = pd.DataFrame(summary_stats)
    # Sort by Kupiec p-value descending (higher is better, meaning closer to theoretical alpha)
    if not df_summary.empty:
        df_summary = df_summary.sort_values(by='Kupiec p-value', ascending=False).reset_index(drop=True)
        
    return df_summary, df_rolling_rates

# Define scenarios to test
models_to_test = [
    {'name': 'Hist_VaR_132d', 'func': historical_var_model, 'kwargs': {'window': 132, 'alpha': 0.01}},
    {'name': 'Hist_VaR_252d', 'func': historical_var_model, 'kwargs': {'window': 252, 'alpha': 0.01}},
    {'name': 'Param_VaR_132d', 'func': parametric_var_model, 'kwargs': {'window': 132, 'alpha': 0.01}},
    {'name': 'Param_VaR_252d', 'func': parametric_var_model, 'kwargs': {'window': 252, 'alpha': 0.01}},
]

summary_table, df_rolling_rates = run_model_comparison(df_returns['log_return'], models_to_test)'''))

cells.append(nbf.v4.new_code_cell('''from IPython.display import display

print("=== Model Comparison Summary ===")
display(summary_table)

plt.figure(figsize=(14, 6))
for col in df_rolling_rates.columns:
    plt.plot(df_rolling_rates.index, df_rolling_rates[col], label=col, alpha=0.8)

plt.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Theoretical Target (1.0%)')
plt.title('Rolling Breach Rate Comparison (252-day moving average)')
plt.ylabel('Breach Rate (%)')
plt.xlabel('Date')
plt.legend()
plt.tight_layout()
plt.show()'''))

nb['cells'] = cells
with open('C:/Users/fe_ma/AFMA_Repos/quant-ai-lab/01_value_at_risk/notebooks/04_modular_var_framework.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
