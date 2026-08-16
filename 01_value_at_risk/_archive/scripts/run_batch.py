import argparse
import json
import os
import shutil
import pandas as pd
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def run_batch_var(tickers, epochs=5, date_report="2026-06-30"):
    # Detect directory structure
    project_root = os.path.dirname(os.path.abspath(__file__))
    notebook_path = os.path.join(project_root, "notebooks", "02_var_straight_forward.ipynb")
    
    if not os.path.exists(notebook_path):
        notebook_path = os.path.join(project_root, "02_var_straight_forward.ipynb")
        if not os.path.exists(notebook_path):
            print(f"Error: Notebook not found at expected path: {notebook_path}")
            return
            
    outputs_dir = os.path.join(project_root, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(os.path.join(outputs_dir, "plots"), exist_ok=True)
    
    # We will build a list to aggregate final summary metrics
    all_summary_data = []
    
    for ticker in tickers:
        print(f"\n" + "="*50)
        print(f" PROCESSING TICKER: {ticker}")
        print("="*50)
        
        # 1. Read notebook template
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
            
        # 2. Modify configuration cell
        conf_cell = None
        for cell in nb.cells:
            if cell.cell_type == "code" and "ticker_widget" in cell.source:
                conf_cell = cell
                break
                
        if conf_cell is None:
            print("Error: Could not find configuration cell in notebook.")
            continue
            
        # Replace parameters in Cell 3
        lines = conf_cell.source.splitlines()
        for idx, line in enumerate(lines):
            if "ticker_widget, date_widget =" in line or "ticker_widget =" in line:
                lines[idx] = f'ticker_widget, date_widget = "{ticker}", "{date_report}"'
            elif "epochs_widget =" in line:
                lines[idx] = f'epochs_widget = {epochs}'
        conf_cell.source = "\n".join(lines)
        
        # 3. Append a cell to save results programmatically
        save_results_code = (
            "# === APPENDED BATCH SAVE CELL ===\n"
            "import os\n"
            "os.makedirs('../outputs', exist_ok=True)\n"
            "df_results.to_json('../outputs/var_results.json')\n"
            "summary_table.to_json('../outputs/var_summary_table.json')\n"
            "print('Successfully saved batch run outputs!')"
        )
        save_cell = nbformat.v4.new_code_cell(source=save_results_code)
        nb.cells.append(save_cell)
        
        # 4. Run notebook using nbconvert's ExecutePreprocessor
        ep = ExecutePreprocessor(timeout=600)
        try:
            print(f"Executing notebook for {ticker}...")
            ep.preprocess(nb, {'metadata': {'path': os.path.join(project_root, "notebooks")}})
            print(f"Finished executing notebook for {ticker}!")
        except Exception as e:
            print(f"Execution failed for {ticker}: {e}")
            continue
            
        # 5. Extract output files
        json_results_path = os.path.join(outputs_dir, "var_results.json")
        json_summary_path = os.path.join(outputs_dir, "var_summary_table.json")
        
        # Locate generated plots
        possible_plot_paths = [
            os.path.join(project_root, "images", "compare_results.png"),
            os.path.join(project_root, "images", "compare_results.pdf"),
            os.path.join(project_root, "outputs", "plots", "compare_results.png")
        ]
        
        found_plot = None
        for path in possible_plot_paths:
            if os.path.exists(path):
                found_plot = path
                break
                
        # Move outputs to ticker-specific filenames
        target_results = os.path.join(outputs_dir, f"{ticker}_var_results.json")
        target_summary = os.path.join(outputs_dir, f"{ticker}_var_summary_table.json")
        
        if os.path.exists(json_results_path):
            shutil.move(json_results_path, target_results)
            print(f"Saved results to {target_results}")
            
        if os.path.exists(json_summary_path):
            try:
                summary_df = pd.read_json(json_summary_path)
                # Reshape summary df to keep Ticker and Model
                summary_df = summary_df.reset_index()
                summary_df.rename(columns={'index': 'Model'}, inplace=True)
                summary_df['Ticker'] = ticker
                all_summary_data.append(summary_df)
            except Exception as e:
                print(f"Error parsing summary table for {ticker}: {e}")
            shutil.move(json_summary_path, target_summary)
            print(f"Saved summary to {target_summary}")
            
        if found_plot:
            ext = os.path.splitext(found_plot)[1]
            target_plot = os.path.join(outputs_dir, "plots", f"{ticker}_compare_results{ext}")
            shutil.copy(found_plot, target_plot)
            print(f"Saved plot to {target_plot}")
            
    # 6. Generate final aggregated summary
    if all_summary_data:
        master_summary = pd.concat(all_summary_data, ignore_index=True)
        # Reorder columns
        cols = ['Ticker', 'Model'] + [col for col in master_summary.columns if col not in ['Ticker', 'Model']]
        master_summary = master_summary[cols]
        
        # Save CSV
        summary_csv_path = os.path.join(outputs_dir, "batch_summary.csv")
        master_summary.to_csv(summary_csv_path, index=False)
        print(f"\nSaved batch summary CSV to {summary_csv_path}")
        
        # Save Markdown
        summary_md_path = os.path.join(outputs_dir, "batch_summary.md")
        with open(summary_md_path, "w", encoding="utf-8") as f:
            f.write("# 🏆 Value-at-Risk Batch Backtesting Summary\n\n")
            # Custom markdown table generator to avoid 'tabulate' package dependency
            headers = list(master_summary.columns)
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
            for _, row in master_summary.iterrows():
                row_str = [str(x) for x in row]
                f.write("| " + " | ".join(row_str) + " |\n")
            f.write("\n")
        print(f"Saved batch summary Markdown to {summary_md_path}")
        
        # Save LaTeX Full Table
        summary_tex_path = os.path.join(outputs_dir, "batch_summary.tex")
        with open(summary_tex_path, "w", encoding="utf-8") as f:
            f.write("% Aggregated Batch Summary Table\n")
            f.write(master_summary.to_latex(index=False))
            f.write("\n")
        print(f"Saved batch summary LaTeX to {summary_tex_path}")
        
        # Save LaTeX Pivoted Tables
        try:
            # Table 1: Safety (Breach Rate and Status)
            master_summary['Safety'] = master_summary['Breach Rate (%)'] + " (" + master_summary['Status'].str.replace('❌', '').str.replace('✅', '').str.strip().str[0] + ")"
            safety_pivot = master_summary.pivot(index='Ticker', columns='Model', values='Safety').reset_index()
            # Fix column names if there's an index name
            safety_pivot.columns.name = None
            
            # Escape % and ^ in the latex output
            safety_tex = safety_pivot.to_latex(index=False).replace('%', r'\%').replace('^', r'\^{}')
            with open(os.path.join(outputs_dir, "batch_summary_safety.tex"), "w", encoding="utf-8") as f:
                f.write("% Safety Pivot Table\n")
                f.write(safety_tex)
                f.write("\n")
                
            # Table 2: Efficiency (Average Capital Reserved)
            eff_pivot = master_summary.pivot(index='Ticker', columns='Model', values='Avg Capital Reserved').reset_index()
            eff_pivot.columns.name = None
            
            eff_tex = eff_pivot.to_latex(index=False).replace('%', r'\%').replace('^', r'\^{}')
            with open(os.path.join(outputs_dir, "batch_summary_efficiency.tex"), "w", encoding="utf-8") as f:
                f.write("% Efficiency Pivot Table\n")
                f.write(eff_tex)
                f.write("\n")
                
            print("Successfully generated pivoted LaTeX tables.")
        except Exception as e:
            print(f"Failed to generate pivoted tables: {e}")
        
        # Extract metrics for LaTeX abstract (using ^GSPC)
        try:
            os.makedirs(os.path.join(outputs_dir, "metrics"), exist_ok=True)
            gspc = master_summary[master_summary['Ticker'] == '^GSPC']
            
            # Find the exact model names
            param_name = [m for m in gspc['Model'] if 'Parametric' in m or 'GARCH' in m][0]
            anchored_name = [m for m in gspc['Model'] if 'Anchor' in m][0]
            
            param_row = gspc[gspc['Model'] == param_name].iloc[0]
            anchored_row = gspc[gspc['Model'] == anchored_name].iloc[0]
            
            paramBreachRate = str(param_row['Breach Rate (%)']).replace('%', '')
            paramMeanVar = str(param_row['Avg Capital Reserved']).replace('%', '')
            
            anchoredBreachRate = str(anchored_row['Breach Rate (%)']).replace('%', '')
            anchoredMeanVar = str(anchored_row['Avg Capital Reserved']).replace('%', '')
            
            capEfficiency = (float(paramMeanVar) - float(anchoredMeanVar)) / float(paramMeanVar) * 100
            
            metrics_to_export = {
                'paramBreaches': int(float(paramBreachRate)/100 * 252), 
                'paramBreachRate': f"{paramBreachRate}\\\\%",
                'paramMeanVar': f"{paramMeanVar}\\\\%",
                'anchoredBreaches': int(float(anchoredBreachRate)/100 * 252),
                'anchoredBreachRate': f"{anchoredBreachRate}\\\\%",
                'anchoredMeanVar': f"{anchoredMeanVar}\\\\%",
                'capEfficiency': f"{capEfficiency:.1f}\\\\%"
            }
            with open(os.path.join(outputs_dir, "metrics", "metrics.tex"), 'w') as f:
                f.write("% Dynamically generated metrics from run_batch.py for ^GSPC\\n")
                for command, value in metrics_to_export.items():
                    f.write(f"\\newcommand{{\\{command}}}{{{value}}}\\n")
            print("Successfully exported LaTeX abstract metrics to outputs/metrics/metrics.tex")
        except Exception as e:
            print(f"Failed to generate abstract metrics.tex: {e}")
        
        print("\n=== BATCH SUMMARY ===")
        print(master_summary.to_string(index=False))
    else:
        print("\nNo summary data was aggregated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VaR analysis batch for multiple tickers.")
    parser.add_argument("--tickers", type=str, default="BTC-USD,^GSPC", help="Comma-separated list of tickers.")
    parser.add_argument("--epochs", type=int, default=3000, help="Number of training epochs.")
    parser.add_argument("--date", type=str, default="2026-06-30", help="Report end date (YYYY-MM-DD).")
    
    args = parser.parse_args()
    ticker_list = [t.strip() for t in args.tickers.split(",") if t.strip()]
    run_batch_var(ticker_list, epochs=args.epochs, date_report=args.date)
