import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# 1. Setup
if not os.path.exists('images'):
    os.makedirs('images')

# --- A. Save the Comparison Plot (The Money Shot) ---
def save_comparison_plot(df, model_1, model_2):
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot Realized
    ax.plot(df.index, df['realized'], color='grey', alpha=0.3, linewidth=1, label='Realized Returns')
    
    # Plot Naive (Failure)
    if model_1 in df.columns:
        ax.plot(df.index, df[model_1], 
                color='firebrick', linestyle='--', linewidth=2, alpha=0.8, label='Naive AI')

    # Plot Hybrid (Success)
    if model_2 in df.columns:
        ax.plot(df.index, df[model_2], 
                color='forestgreen', linewidth=2.5, label='Physics-Informed AI')

    # Styling
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.set_title(f'Deep VaR: {model_1} vs. {model_2}', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('Log Returns / VaR')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Focus on Risk (Negative)
    ax.set_ylim(top=0.0)
    ax.axhline(0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(f'images/{model_1}_vs_{model_2}.png', dpi=300)
    print(f"✅ Saved images/{model_1}_vs_{model_2}.png")

# --- B. Save the Summary Table as an Image ---
def save_table_image(df_table):
    # # Re-creating the data from your latest results
    # data = {
    #     "Model": ["1. Just Train (Naive)", "2. Back to Basics", "3. Feature Eng.", "4. Physics-Informed (Hybrid)", "6. Historical VaR"],
    #     "Breach Rate": ["0.14%", "0.14%", "0.57%", "1.14%", "1.57%"],
    #     "Capital Reserved": ["-10.32%", "-10.39%", "-7.11%", "-6.49%", "-6.22%"],
    #     "Responsiveness": ["0.0082", "0.0030", "0.0185", "0.0101", "0.0054"],
    #     "Status": ["❌ FAIL", "❌ FAIL", "✅ PASS", "✅ WINNER", "✅ PASS"]
    # }
    # df_table = pd.DataFrame(data)

    # Render as plot
    fig, ax = plt.subplots(figsize=(10, 4)) 
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df_table.values, colLabels=df_table.columns, loc='center', cellLoc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2) # Adjust spacing
    
    # Color the header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#40466e')
            cell.set_text_props(color='white', weight='bold')
        if row > 0 and df_table.iloc[row-1]['Status'] == "✅ WINNER":
             cell.set_facecolor('#e6fffa') # Highlight the winner

    plt.savefig('images/summary_table.png', dpi=300, bbox_inches='tight')
    print("✅ Saved images/summary_table.png")

# save_table_image()