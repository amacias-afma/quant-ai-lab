import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def generate_plots():
    # Load data
    df = pd.read_csv('outputs/showdown_results_static.csv')
    
    # Drop empty rows
    df = df.dropna(subset=['Ticker'])
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Showdown Results: Classical Baselines vs Wide & Deep ML', fontsize=16, fontweight='bold')
    
    # Plot 1: NLL
    width = 0.35
    x = np.arange(len(df['Ticker']))
    
    axes[0].bar(x - width/2, df['NLL_Baseline'], width, label='Classical Baseline', color='#1f77b4')
    axes[0].bar(x + width/2, df['NLL_ML'], width, label='Wide & Deep ML', color='#ff7f0e')
    axes[0].set_ylabel('Negative Log-Likelihood (Lower is Better)')
    axes[0].set_title('Negative Log-Likelihood Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df['Ticker'], rotation=45)
    axes[0].legend()
    
    # Plot 2: CRPS
    axes[1].bar(x - width/2, df['CRPS_Baseline'], width, label='Classical Baseline', color='#1f77b4')
    axes[1].bar(x + width/2, df['CRPS_ML'], width, label='Wide & Deep ML', color='#ff7f0e')
    axes[1].set_ylabel('CRPS (Lower is Better)')
    axes[1].set_title('Continuous Ranked Probability Score Comparison')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df['Ticker'], rotation=45)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(r'C:\Users\fe_ma\.gemini\antigravity\brain\5843fdfb-5d5a-4e9b-9f42-07a0115ed738\showdown_plot.png', dpi=300)
    print("Plot saved successfully.")

if __name__ == '__main__':
    generate_plots()
