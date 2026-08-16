import os
import json
import pandas as pd
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from value_at_risk.utils.plot_utils import plot_var_results

def main():
    outputs_dir = os.path.join(os.path.dirname(__file__), "outputs")
    plots_dir = os.path.join(outputs_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # List of tickers to process
    tickers = ["BTC-USD", "CL=F", "CLP=X", "HG=F", "NVDA", "SQM", "TSLA", "^GSPC"]
    
    all_summary_data = []
    
    for ticker in tickers:
        print(f"Processing {ticker}...")
        results_file = os.path.join(outputs_dir, f"{ticker}_var_results.json")
        summary_file = os.path.join(outputs_dir, f"{ticker}_var_summary_table.json")
        
        if not os.path.exists(results_file) or not os.path.exists(summary_file):
            print(f"Missing data for {ticker}, skipping.")
            continue
            
        # 1. Regenerate Plot
        df_results = pd.read_json(results_file)
        
        plot_path = f"../outputs/plots/{ticker}_compare_results.png"
        plot_var_results(df_results, file_name=plot_path, show=False)
        print(f"  Saved plot for {ticker}")
        
        # 2. Collect Summary Data
        summary_df = pd.read_json(summary_file)
        summary_df = summary_df.reset_index()
        summary_df.rename(columns={'index': 'Model'}, inplace=True)
        summary_df['Ticker'] = ticker
        all_summary_data.append(summary_df)

    # 3. Generate Aggregated Summaries
    if all_summary_data:
        master_summary = pd.concat(all_summary_data, ignore_index=True)
        cols = ['Ticker', 'Model'] + [col for col in master_summary.columns if col not in ['Ticker', 'Model']]
        master_summary = master_summary[cols]
        
        # Save CSV
        master_summary.to_csv(os.path.join(outputs_dir, "batch_summary.csv"), index=False)
        
        # Save LaTeX Full
        with open(os.path.join(outputs_dir, "batch_summary.tex"), "w", encoding="utf-8") as f:
            f.write("% Aggregated Batch Summary Table\n")
            f.write(master_summary.to_latex(index=False))
            f.write("\n")
            
        # Table 1: Safety
        master_summary['Safety'] = master_summary['Breach Rate (%)'] + " (" + master_summary['Status'].str.replace('❌', '').str.replace('✅', '').str.strip().str[0] + ")"
        safety_pivot = master_summary.pivot(index='Ticker', columns='Model', values='Safety').reset_index()
        safety_pivot.columns.name = None
        safety_tex = safety_pivot.to_latex(index=False).replace('%', r'\%').replace('^', r'\^{}')
        with open(os.path.join(outputs_dir, "batch_summary_safety.tex"), "w", encoding="utf-8") as f:
            f.write("% Safety Pivot Table\n")
            f.write(safety_tex)
            f.write("\n")
            
        # Table 2: Efficiency
        eff_pivot = master_summary.pivot(index='Ticker', columns='Model', values='Avg Capital Reserved').reset_index()
        eff_pivot.columns.name = None
        eff_tex = eff_pivot.to_latex(index=False).replace('%', r'\%').replace('^', r'\^{}')
        with open(os.path.join(outputs_dir, "batch_summary_efficiency.tex"), "w", encoding="utf-8") as f:
            f.write("% Efficiency Pivot Table\n")
            f.write(eff_tex)
            f.write("\n")
            
        print("Generated batch summaries and pivoted LaTeX tables.")

if __name__ == "__main__":
    main()
