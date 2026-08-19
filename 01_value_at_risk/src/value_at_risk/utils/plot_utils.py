import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy import stats
import os
from value_at_risk.evaluation.backtest_value_at_risk import backtest_var_models

def plot_var_comparison(df_results, alpha=0.05):
    """
    Create comprehensive comparison plots for VaR models.
    """
    # Clean up index if needed
    plot_index = pd.to_datetime([idx[0] if isinstance(idx, tuple) else idx 
                                  for idx in df_results.index])
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: VaR estimates comparison
    ax1 = fig.add_subplot(gs[0, :])
    
    # Plot realized returns
    breaches_nn = df_results['realized'] < df_results['predicted']
    breaches_param = df_results['realized'] < df_results['parametric']
    
    # Points that breach both models (red)
    both_breach = breaches_nn & breaches_param
    # Points that breach only NN (orange)
    only_nn = breaches_nn & ~breaches_param
    # Points that breach only Parametric (yellow)
    only_param = ~breaches_nn & breaches_param
    # No breaches (green)
    no_breach = ~breaches_nn & ~breaches_param
    
    ax1.scatter(plot_index[no_breach], df_results.loc[no_breach, 'realized'],
               color='green', s=20, alpha=0.6, label='No Breach', zorder=2)
    ax1.scatter(plot_index[only_param], df_results.loc[only_param, 'realized'],
               color='yellow', s=30, alpha=0.8, label='Parametric Breach Only', zorder=3)
    ax1.scatter(plot_index[only_nn], df_results.loc[only_nn, 'realized'],
               color='orange', s=30, alpha=0.8, label='NN Breach Only', zorder=3)
    ax1.scatter(plot_index[both_breach], df_results.loc[both_breach, 'realized'],
               color='red', s=40, alpha=0.9, label='Both Breach', zorder=4)
    
    # Plot VaR lines
    ax1.plot(plot_index, df_results['predicted'], 
            color='blue', linewidth=2, label='Neural Network VaR', linestyle='--', alpha=0.7)
    ax1.plot(plot_index, df_results['parametric'], 
            color='purple', linewidth=2, label='Parametric VaR', linestyle='-.', alpha=0.7)
    
    ax1.set_ylabel('Returns', fontsize=11)
    ax1.set_title(f'VaR Model Comparison: Realized Returns vs Predictions (α={alpha})', 
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='lower left', fontsize=9, ncol=3)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 2: Cumulative breach rates - Neural Network
    ax2 = fig.add_subplot(gs[1, 0])
    
    breaches_nn_cum = breaches_nn.expanding().sum()
    count_cum = breaches_nn.expanding().count()
    breach_rate_nn = breaches_nn_cum / count_cum
    
    rolling = 252
    plot_index = plot_index[rolling:]
    breach_rate_nn = breach_rate_nn[rolling:]
    
    # print(plot_index)
    # print(breach_rate_nn)

    ax2.plot(plot_index, breach_rate_nn * 100, 
            color='blue', linewidth=2, label='Neural Network')
    ax2.axhline(y=alpha * 100, color='red', linestyle='--', linewidth=2, 
               label=f'Expected ({alpha*100:.1f}%)')
    ax2.fill_between(plot_index, 0, breach_rate_nn * 100, alpha=0.3, color='blue')
    
    final_rate_nn = breach_rate_nn.iloc[-1] * 100
    ax2.text(0.02, 0.98, f'Final Rate: {final_rate_nn:.2f}%',
            transform=ax2.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Cumulative Breach Rate (%)', fontsize=11)
    ax2.set_title('Neural Network VaR', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 3: Cumulative breach rates - Parametric
    ax3 = fig.add_subplot(gs[1, 1])
    
    breaches_param_cum = breaches_param.expanding().sum()
    breach_rate_param = breaches_param_cum / count_cum
    breach_rate_param = breach_rate_param[rolling:]
    
    ax3.plot(plot_index, breach_rate_param * 100, 
            color='purple', linewidth=2, label='Parametric')
    ax3.axhline(y=alpha * 100, color='red', linestyle='--', linewidth=2, 
               label=f'Expected ({alpha*100:.1f}%)')
    ax3.fill_between(plot_index, 0, breach_rate_param * 100, alpha=0.3, color='purple')
    
    final_rate_param = breach_rate_param.iloc[-1] * 100
    ax3.text(0.02, 0.98, f'Final Rate: {final_rate_param:.2f}%',
            transform=ax3.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='plum', alpha=0.7))
    
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_ylabel('Cumulative Breach Rate (%)', fontsize=11)
    ax3.set_title('Parametric VaR (Normal)', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 4: Comparison bar chart
    ax4 = fig.add_subplot(gs[2, :])

    stats_models = backtest_var_models(df_results, alpha=alpha)

    models = list(stats_models.keys())
    breach_rates = [stats_models[m]['breach_rate'] * 100 for m in models]
    expected = alpha * 100

    x = np.arange(len(models))
    width = 0.6  # Make bars wider since we only have one set

    # Plot actual breach rates as bars
    bars = ax4.bar(x, breach_rates, width, label='Actual Breach Rate',
                color=['blue', 'purple'], alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add expected rate as a horizontal reference line
    ax4.axhline(y=expected, color='red', linestyle='--', linewidth=2.5, 
                label=f'Expected Rate ({expected:.1f}%)', zorder=10)

    ax4.set_ylabel('Breach Rate (%)', fontsize=11)
    ax4.set_title('VaR Model Performance Comparison', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models, fontsize=10)
    ax4.legend(fontsize=10, loc='best')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, rate) in enumerate(zip(bars, breach_rates)):
        height = bar.get_height()
        # Show the value
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Show the difference from expected
        diff = rate - expected
        color = 'green' if abs(diff) < 1 else 'orange' if abs(diff) < 2 else 'red'
        ax4.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{diff:+.2f}%', ha='center', va='center', fontsize=8, 
                color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Set y-axis limits to give some space
    y_max = max(max(breach_rates), expected) * 1.2
    ax4.set_ylim(0, y_max)

    plt.tight_layout()
    return fig, stats_models

def improved_price_plot(df, ticker_name, column='price'):
        
    # Create an improved price plot
    fig, ax = plt.subplots(figsize=(14, 6))
    units = {'price': '$', 'log_ret': ''}
    # Plot the price data with styling
    ax.plot(df.index, df[column], linewidth=2, color='steelblue', label=f'{ticker_name} {column.capitalize()}')

    # Format x-axis with dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # Show every 6 months
    plt.xticks(rotation=45, ha='right')

    # Add labels and title
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{column} {units[column]}', fontsize=12, fontweight='bold')
    ax.set_title(f'{ticker_name} - Historical {column.capitalize()}', 
                fontsize=14, fontweight='bold', pad=20)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Add legend
    ax.legend(loc='best', fontsize=10)

    # Add statistics box
    stats_text = f'Min: ${df[column].min():.2f}\nMax: ${df[column].max():.2f}\nMean: ${df[column].mean():.2f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

def plot_var_results(df_results, models=None, file_name=None, show=True):
        # Reset the index to get clean datetime values
        df_plot = df_results.copy()
        
        # Clean up index if it's a multi-index or contains tuples
        df_plot.index = pd.to_datetime([idx[0] if isinstance(idx, tuple) else idx for idx in df_plot.index])
        
        if models is None:
            # Default to all columns that are not 'realized' or 'date'
            models = [col for col in df_plot.columns if col not in ['realized', 'date']]
        elif isinstance(models, set):
            models = list(models)
            
        # Sort models so the order is deterministic
        models = sorted(models)
        num_models = len(models)
        
        # Create subplots: 2 columns if 4 or more models, otherwise 1 column
        import math
        cols = 2 if num_models >= 4 else 1
        rows = math.ceil(num_models / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows), sharex=True, sharey=True)
        
        if num_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
            
        colors = ['#1f77b4', '#9467bd', '#2ca02c', '#d62728', '#ff7f0e', '#17becf', '#e377c2', '#8c564b', '#7f7f7f', '#bcbd22']
        
        for i, model in enumerate(models):
            ax = axes[i]
            color = colors[i % len(colors)]
            
            # Plot realized returns in a light gray line for context
            ax.plot(df_plot.index, df_plot['realized'], color='#d3d3d3', alpha=0.5, linewidth=1, label='Realized Returns')
            
            # Plot model VaR line
            ax.plot(df_plot.index, df_plot[model], color=color, linewidth=2, linestyle='--', label=f'{model} VaR')
            
            # Calculate and plot breaches for this model specifically
            breaches = df_plot['realized'] < df_plot[model]
            non_breaches = ~breaches
            
            # Plot non-breaches in light green
            ax.scatter(df_plot[non_breaches].index, df_plot[non_breaches]['realized'], 
                       color='#2ca02c', s=15, alpha=0.3, label='No Breach', zorder=2)
            
            # Highlight breaches in vivid red
            ax.scatter(df_plot[breaches].index, df_plot[breaches]['realized'], 
                       color='#d62728', s=25, alpha=0.9, label='Breach', zorder=3, edgecolors='black', linewidth=0.5)
            
            # Calculate statistics
            total_obs = len(df_plot)
            num_breaches = int(breaches.sum())
            breach_rate = num_breaches / total_obs
            mean_var = abs(df_plot[model].mean())
            
            # Kupiec p-value
            try:
                p_val = stats.binomtest(num_breaches, total_obs, 0.01, alternative='two-sided').pvalue
            except AttributeError:
                p_val = stats.binom_test(num_breaches, total_obs, 0.01, alternative='two-sided')
                
            status = "PASS" if p_val > 0.05 and abs(breach_rate - 0.01) < 0.01 else "FAIL"
            
            # Text box with key metrics
            stats_text = (
                f"{model}\n"
                f"Breaches: {num_breaches}/{total_obs} ({breach_rate*100:.2f}%)\n"
                f"Mean VaR: {mean_var*100:.2f}%\n"
                f"Kupiec p-val: {p_val:.4f} ({status})"
            )
            
            ax.text(0.01, 0.95, stats_text, transform=ax.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#e2e8f0', alpha=0.9))
            
            ax.set_ylabel('Returns / VaR', fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
            ax.set_title(f"Model Comparison: {model}", fontsize=11, fontweight='bold', loc='left')

        # Hide unused subplots
        for j in range(num_models, len(axes)):
            fig.delaxes(axes[j])
            
        # Format x-axis dates for all active axes
        for i in range(num_models):
            axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            axes[i].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            axes[i].tick_params(axis='x', rotation=45, labelsize=10)
            
            # Only set xlabel on bottom row
            if i >= num_models - cols:
                axes[i].set_xlabel('Date', fontsize=12)
        
        # Set y-limit to focus on negative returns and VaR
        all_min = min(df_plot['realized'].min(), df_plot[models].min().min())
        plt.ylim(bottom=all_min * 1.1, top=0.02)
        
        plt.suptitle('Value-at-Risk (VaR) Backtesting Comparison (α = 1%)', fontsize=14, fontweight='bold', y=0.99)
        plt.tight_layout()
        
        if file_name:
            if os.path.isabs(file_name) or file_name.startswith('..'):
                save_target = file_name
            else:
                save_target = f'../images/{file_name}'
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(save_target)), exist_ok=True)
            plt.savefig(save_target, dpi=300, bbox_inches='tight')
            
        if show:
            plt.show()

import matplotlib.dates as mdates

def compare_two_models(df_results, model_1, model_2):
    # Create the Focused Plot (Negative Only)
    plt.figure(figsize=(15, 7))

    # 1. Plot Realized Returns (The Chaos)
    # We plot everything, but we will crop the view later
    plt.plot(df_results.index, df_results['realized'], 
            color='gray', alpha=0.3, linewidth=1, label='Realized Returns')

    # 2. First model (typically the unanchored ablation)
    plt.plot(df_results.index, df_results[model_1],
            color='red', linestyle='--', linewidth=2, alpha=0.7,
            label=model_1)

    # 3. Second model (typically the anchored model)
    plt.plot(df_results.index, df_results[model_2], 
            color='green', linewidth=2.5, 
            label=model_2)

    # Formatting
    plt.title(f"Comparing two models: {model_1} v/s {model_2} (Negative Returns Only)", fontsize=14, fontweight='bold')
    plt.ylabel("Log Returns / VaR", fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.legend(loc='lower left', frameon=True, fontsize=11)
    plt.grid(True, alpha=0.2)

    # --- THE TRICK: Focus on Negative Side ---
    plt.ylim(top=0.0) # Cut off positive returns
    plt.axhline(0, color='black', linewidth=0.5) # Add a zero line

    # Format x-axis dates nicely
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()



def plot_breach_rate(df_results, alpha=0.01):
    # Calculate breach statistics
    df_results_breach = df_results.copy()
    df_results_breach['breach'] = (df_results_breach['realized'] < df_results_breach['predicted']).astype(int)

    # Calculate rolling breach rate (e.g., over 60-day window)
    window = 252
    df_results_breach['rolling_breach'] = df_results_breach['breach'].rolling(window=window).sum()
    df_results_breach['rolling_breach_count'] = df_results_breach['breach'].rolling(window=window).count()
    df_results_breach['breach_rate'] = df_results_breach['rolling_breach'] / df_results_breach['rolling_breach_count']

    # Calculate cumulative breach rate
    df_results_breach['cumulative_breach'] = df_results_breach['breach'].expanding().sum()
    df_results_breach['cumulative_count'] = df_results_breach['breach'].expanding().count()
    df_results_breach['cumulative_breach_rate'] = df_results_breach['cumulative_breach'] / df_results_breach['cumulative_count']

    df_results_breach.dropna(inplace=True)    
    # Create improved plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Clean up index if needed (remove tuples)
    plot_index = pd.to_datetime([idx[0] if isinstance(idx, tuple) else idx for idx in df_results_breach.index])

    # Plot 1: Rolling breach rate
    ax1 = axes[0]
    ax1.plot(plot_index, df_results_breach['breach_rate'] * 100, 
            color='steelblue', linewidth=2, label=f'{window}-day Rolling Breach Rate')
    ax1.axhline(y=alpha * 100, color='red', linestyle='--', linewidth=2, 
                label=f'Expected Rate ({alpha*100:.1f}%)')
    ax1.fill_between(plot_index, 0, df_results_breach['breach_rate'] * 100, 
                    alpha=0.3, color='steelblue')

    ax1.set_ylabel('Breach Rate (%)', fontsize=12)
    ax1.set_title(f'VaR Breach Rate Analysis (α={alpha})', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 2: Cumulative breach rate
    ax2 = axes[1]
    ax2.plot(plot_index, df_results_breach['cumulative_breach_rate'] * 100, 
            color='darkgreen', linewidth=2, label='Cumulative Breach Rate')
    ax2.axhline(y=alpha * 100, color='red', linestyle='--', linewidth=2, 
                label=f'Expected Rate ({alpha*100:.1f}%)')
    ax2.fill_between(plot_index, 0, df_results_breach['cumulative_breach_rate'] * 100, 
                    alpha=0.3, color='darkgreen')

    # Add final statistics as text
    final_rate = df_results_breach['cumulative_breach_rate'].iloc[-1] * 100
    total_breaches = int(df_results_breach['cumulative_breach'].iloc[-1])
    total_obs = int(df_results_breach['cumulative_count'].iloc[-1])

    ax2.text(0.02, 0.98, f'Final Statistics:\nTotal Breaches: {total_breaches}/{total_obs}\nBreach Rate: {final_rate:.2f}%',
            transform=ax2.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Cumulative Breach Rate (%)', fontsize=12)
    ax2.set_title('Cumulative VaR Breach Rate', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("=" * 60)
    print("VaR BACKTESTING SUMMARY")
    print("=" * 60)
    print(f"Expected Breach Rate (α): {alpha*100:.2f}%")
    print(f"Actual Breach Rate: {final_rate:.2f}%")
    print(f"Total Breaches: {total_breaches} out of {total_obs} observations")
    print(f"Difference from Expected: {final_rate - alpha*100:+.2f}%")
    print("=" * 60)