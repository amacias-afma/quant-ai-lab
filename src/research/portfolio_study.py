import os
import sys
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from google.cloud import bigquery
from arch import arch_model

# Add project root to sys.path to allow imports from src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.evaluation.backtest import backtest
from data.market_data import MarketData

# Configuration
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "quant-ai-lab")
PORTFOLIO_SIZE = 1_000_000
CONFIDENCE_LEVEL = 2.33 # 99% Normal

SCENARIOS = {
    "Covid_Crash": ("2020-01-01", "2020-06-30"), # The main stress test
    "Inflation_Bear": ("2022-01-01", "2022-12-31"),
    "Volmageddon (2018)":  ("2018-01-01", "2018-06-01"),
    "Year 2024": ("2024-01-01", "2024-12-31"),
    "Year 2025": ("2025-01-01", "2025-12-31")
}

# The "Diversified 9"
ASSETS = [
    "^GSPC", "BTC-USD", "CLP=X",  # S&P500, Bitcoin, Chile Peso
    "SQM", "HG=F", "TSLA",        # Lithium, Copper, Tesla
    "NVDA", "CL=F", "TLT"         # Nvidia, Oil, Treasuries
]

# --- Main Study Class ---
class PortfolioStudy:
    def __init__(self):
        self.client = bigquery.Client(project=PROJECT_ID)
        self.results = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_data(self, ticker):
        query = f"""
            SELECT date, log_ret 
            FROM `{PROJECT_ID}.market_data.historical_prices`
            WHERE ticker = '{ticker}' ORDER BY date ASC
        """
        df = self.client.query(query).to_dataframe()
        df['date'] = pd.to_datetime(df['date'])
        return df.set_index('date').dropna()

    def train_lstm(self, train_series):
        # Prepare Data
        X, y = prepare_lstm_data(train_series)
        model = VolatilityLSTM().to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        criterion = Log_QLIKE_Loss()
        
        # Fast Training (50 epochs is enough for demonstration)
        model.train()
        for _ in range(50):
            optimizer.zero_grad()
            out = model(X.to(self.device)).squeeze()
            loss = criterion(out, y.to(self.device))
            loss.backward()
            optimizer.step()
            
        return model

    def run_scenario(self, ticker, scenario_name, dates):
        print(f"  > Analyzing {ticker} for {scenario_name}...")

        market_data = MarketData(ticker, project_id='quant-ai-lab')
        market_data.load_data()
        df = market_data.data
        df['log_ret'] = np.log(df['price'] / df['price'].shift(1))
        df = df.dropna()
        start_date, end_date = dates
        
        # print(df.head())
        # 1. Split Data
        train_mask = df.index < start_date
        test_mask = (df.index >= start_date) & (df.index <= end_date)
        
        train_data = df.loc[train_mask, 'log_ret']
        test_data = df.loc[test_mask, 'log_ret']
        
        if len(test_data) < 10: return None # Skip if no data

        # 2. Train Models (On Pre-Crisis Data)
        # Configuration for VaR
        portfolio_value = 1_000_000
        confidence_level = 2.33 # 99% Confidence
        res_bt = backtest(portfolio_value, confidence_level, df, start_date, end_date, scenario_name, visual=False)
        var_garch = res_bt["var_garch"]
        var_hist = res_bt["var_hist"]
        var_lstm = res_bt["var_lstm"]
        
        daily_loss = res_bt["daily_loss"]
        dates = res_bt["test_data"].index

        b_hist = float((daily_loss > var_hist).mean() * 100)
        b_garch = float((daily_loss > var_garch).mean() * 100)
        b_lstm = float((daily_loss > var_lstm).mean() * 100)
        # Also ensure these are floats
        cap_garch = float(var_garch.mean())
        cap_lstm = float(var_lstm.mean())
        
        timeseries_data = {
            "dates": test_data.index.strftime('%Y-%m-%d').tolist(),
            "loss": daily_loss.fillna(0).tolist(),
            "var_hist": var_hist.fillna(0).tolist(),
            "var_garch": var_garch.fillna(0).tolist(),
            "var_lstm": var_lstm.fillna(0).tolist()
        }
        return {
            "ticker": ticker,
            "scenario": scenario_name,
            "hist_breach": b_hist,
            "garch_breach": b_garch,
            "lstm_breach": b_lstm,
            "garch_capital": cap_garch,
            "lstm_capital": cap_lstm,
            "capital_savings": cap_garch - cap_lstm, # Positive = LSTM Saved Money
            "timeseries_json": json.dumps(timeseries_data)
            # "data": { # Save series for plotting
            #     "dates": test_data.index,
            #     "loss": daily_loss,
            #     "var_hist": var_hist,
            #     "var_garch": var_garch,
            #     "var_lstm": var_lstm
            # },
            # "timeseries_data": timeseries_data
        }

    def run_full_study(self):
        print(f"Starting Study: {len(ASSETS)} Assets x {len(SCENARIOS)} Scenarios...")
        
        for name, dates in SCENARIOS.items():
            for ticker in ASSETS:
                try:
                    res = self.run_scenario(ticker, name, dates)
                    if res: self.results.append(res)
                except Exception as e:
                    print(f"Error {ticker}: {e}")

        # Save to BigQuery
        df = pd.DataFrame(self.results)
        # print('results')
        # print(df.head())
        
        table_id = f"{PROJECT_ID}.market_data.research_distribution"
        df.to_gbq(table_id, project_id=PROJECT_ID, if_exists='replace')
        print("✅ Study Complete. Data saved to BigQuery.")

        # print(f"Starting Multi-Asset Study ({len(ASSETS)} assets)...")
        
        # # Focus on COVID for the visual grid
        # scenario = "Covid_Crash"
        # dates = SCENARIOS[scenario]
        
        # plot_data = []
        
        # for ticker in ASSETS:
        #     res = self.run_scenario(ticker, scenario, dates)
        #     if res:
        #         self.results.append(res)
        #         plot_data.append(res)
        
        # # Generate 3x3 Plot
        # self.plot_grid(plot_data, scenario)
        
        # # Save Summary to BigQuery
        # self.save_summary()

    def plot_grid(self, data_list, title_suffix):
        """Generates a 3x3 grid of Capital Shields"""
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        fig.suptitle(f"Capital Shield Stress Test: {title_suffix} (GARCH vs Deep LSTM)", fontsize=16)
        
        for i, ax in enumerate(axes.flat):
            if i >= len(data_list): break
            d = data_list[i]
            
            # Unpack series
            dates = d['Data']['Dates']
            loss = d['Data']['Loss']
            # Filter loss for cleaner plot
            loss = loss.where(loss > 0, 0)
            
            ax.bar(dates, loss, color='gray', alpha=0.3, label='Loss')
            ax.plot(dates, d['Data']['VaR_GARCH'], 'b-', alpha=0.6, label='GARCH')
            ax.plot(dates, d['Data']['VaR_LSTM'], 'r-', linewidth=1.5, label='LSTM (AI)')
            
            # # Title with Metrics
            # print('----------------') 
            # print(d)
            # print('----------------') 

            # ax.set_title(f"{d['Ticker']}\nLSTM Breach: {d['LSTM_Breach']:.1f}% | GARCH: {d['GARCH_Breach']:.1f}%")

            # ax.set_title(f"{d['Ticker']}\nLSTM Breach: {d['LSTM_Breach']:.1f}% | GARCH: {d['GARCH_Breach']:.1f}%")
            ax.grid(True, alpha=0.2)
            
            if i == 0: ax.legend() # Only legend on first
            
        plt.tight_layout()
        plt.savefig("portfolio_stress_test.png")
        print("✅ Plot saved to portfolio_stress_test.png")
    
    def save_summary(self):
        # Convert results to DataFrame for BigQuery
        rows = []
        for r in self.results:
            rows.append({
                "ticker": str(r['Ticker']), # Ensure string
                "scenario": str(r['Scenario']),
                "hist_breach": float(r['Hist_Breach']), # Ensure float
                "garch_breach": float(r['GARCH_Breach']),
                "lstm_breach": float(r['LSTM_Breach']),
                "capital_savings": float(r['Capital_Savings'])
            })
        
        df = pd.DataFrame(rows)
        # Force numeric types
        df['hist_breach'] = pd.to_numeric(df['hist_breach'])
        df['garch_breach'] = pd.to_numeric(df['garch_breach'])
        df['lstm_breach'] = pd.to_numeric(df['lstm_breach'])
        df['capital_savings'] = pd.to_numeric(df['capital_savings'])

        # Upload to BigQuery (Replace table)
        table_id = f"{PROJECT_ID}.market_data.research_results"
        # Schema definition helps BigQuery understand the types
        job_config = bigquery.LoadJobConfig(
            schema=[
                bigquery.SchemaField("ticker", "STRING"),
                bigquery.SchemaField("scenario", "STRING"),
                bigquery.SchemaField("hist_breach", "FLOAT"),
                bigquery.SchemaField("garch_breach", "FLOAT"),
                bigquery.SchemaField("lstm_breach", "FLOAT"),
                bigquery.SchemaField("capital_savings", "FLOAT"),
            ],
            write_disposition="WRITE_TRUNCATE",
        )

        self.client.load_table_from_dataframe(df, table_id, job_config=job_config).result()
        print("✅ Research Results uploaded to BigQuery.")

    # def save_summary(self):
    #     # Convert results to DataFrame for BigQuery
    #     rows = []
    #     for r in self.results:
    #         rows.append({
    #             "ticker": r['Ticker'],
    #             "scenario": r['Scenario'],
    #             "hist_breach": r['Hist_Breach'],
    #             "garch_breach": r['GARCH_Breach'],
    #             "lstm_breach": r['LSTM_Breach']
    #             # "capital_savings": r['Capital_Savings']
    #         })
        
    #     df = pd.DataFrame(rows)
    #     # Upload to BigQuery (Replace table)
    #     table_id = f"{PROJECT_ID}.market_data.research_results"
    #     df.to_gbq(table_id, project_id=PROJECT_ID, if_exists='replace')
    #     print("✅ Research Results uploaded to BigQuery.")

if __name__ == "__main__":
    study = PortfolioStudy()
    study.run_full_study()