import pandas as pd
import numpy as np
import os
import json
from google.cloud import bigquery
from baseline import GarchBaseline
from lstm import DeepVolEngine

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "quant-ai-lab")
DATASET_ID = "market_data"
INPUT_TABLE = "historical_prices"
OUTPUT_TABLE = "dashboard_forecasts"

PORTFOLIO_SIZE = 1_000_000 
VAR_CONFIDENCE = 2.33 # Z-score for Normal Dist (GARCH/LSTM)
HISTORICAL_QUANTILE = 0.01 # 1% Quantile for Historical Sim

TICKER_MAP = {
    "^GSPC": "S&P 500 Index",
    "BTC-USD": "Bitcoin",
    "CLP=X": "USD/CLP (Chilean Peso)",
    "SQM": "SQM (Lithium Chile)",
    "HG=F": "Copper Futures",
    "TSLA": "Tesla Inc.",
    "NVDA": "NVIDIA Corp.",
    "CL=F": "Crude Oil WTI",
    "TLT": "US Treasury Bond 20Y",
    "VXX": "VIX Short-Term Futures"
}

class BatchRiskEngine:
    def __init__(self):
        self.client = bigquery.Client(project=PROJECT_ID)

    def load_ticker_data(self, ticker):
        query = f"""
            SELECT date, log_ret, target_variance 
            FROM `{PROJECT_ID}.{DATASET_ID}.{INPUT_TABLE}`
            WHERE ticker = '{ticker}'
            ORDER BY date ASC
        """
        df = self.client.query(query).to_dataframe()
        if not df.empty:
            df.set_index('date', inplace=True)
        return df

    def calculate_metrics(self, returns, capital_series, model_name):
        """
        Calculates VaR Backtest metrics.
        """
        # Align lengths (last 100 days)
        n = len(returns)
        cap = np.array(capital_series[-n:])
        ret = np.array(returns[-n:]) * PORTFOLIO_SIZE
        
        # Loss = Positive magnitude of negative returns
        losses = -ret
        
        # Breach: When Loss > Capital Held
        breaches = losses > cap
        num_breaches = int(np.sum(breaches))
        breach_rate = (num_breaches / n) * 100 # In percentage
        
        avg_cap = float(np.mean(cap))
        
        return {
            "series": cap.tolist(),
            "breaches": num_breaches,
            "rate": breach_rate,
            "avg_cap": avg_cap
        }

    def run_batch(self):
        print(f"🚀 Starting Validator Batch Run...")
        results = []

        for ticker, name in TICKER_MAP.items():
            print(f"   ... Processing {name}")
            try:
                df = self.load_ticker_data(ticker)
                # Need 252 days for Historical Window + 100 days for Plot = 352
                if len(df) < 360: continue 

                # --- 1. MODELING ---
                
                # A. HISTORICAL SIMULATION (The "Dumb" Benchmark)
                # Rolling 1-year (252 days) quantile of returns * Portfolio
                # Note: quantile(0.01) gives negative return. We want positive capital requirement.
                hist_var_rolling = df['log_ret'].rolling(252).quantile(HISTORICAL_QUANTILE)
                # Convert to positive capital amount: abs(return) * Portfolio
                hist_capital_full = hist_var_rolling.abs() * PORTFOLIO_SIZE
                
                # B. GARCH
                garch = GarchBaseline(ticker, data=df)
                garch.train()
                # GARCH Vol * Z-Score * Portfolio
                garch_sigma = garch.model_fit.conditional_volatility / 100 # Decimals
                garch_capital_full = garch_sigma * VAR_CONFIDENCE * PORTFOLIO_SIZE
                
                # C. LSTM
                lstm = DeepVolEngine(ticker, seq_len=60, data=df)
                lstm.train(epochs=15)
                pred_vol, _ = lstm.predict(df) 
                # LSTM Vol (Ann) -> Daily Vol -> Capital
                lstm_sigma_daily = (pred_vol / np.sqrt(252))
                lstm_capital_full = lstm_sigma_daily * VAR_CONFIDENCE * PORTFOLIO_SIZE

                # --- 2. BACKTEST (Last 100 Days) ---
                eval_window = 100
                returns_slice = df['log_ret'][-eval_window:].tolist()
                
                # Get metrics for all 3 models
                m_hist = self.calculate_metrics(returns_slice, hist_capital_full, "Historical")
                m_garch = self.calculate_metrics(returns_slice, garch_capital_full, "GARCH")
                m_lstm = self.calculate_metrics(returns_slice, lstm_capital_full, "LSTM")

                # --- 3. PACKAGING ---
                results.append({
                    "ticker": ticker,
                    "asset_name": name,
                    "last_updated": pd.Timestamp.now(),
                    "dates_json": json.dumps(df.index[-eval_window:].strftime('%Y-%m-%d').tolist()),
                    "returns_json": json.dumps([x * PORTFOLIO_SIZE for x in returns_slice]),
                    
                    # Series for Plotting
                    "hist_var_json": json.dumps(m_hist['series']),
                    "garch_var_json": json.dumps(m_garch['series']),
                    "lstm_var_json": json.dumps(m_lstm['series']),
                    
                    # Scalar Metrics for Table
                    "hist_breaches": m_hist['breaches'],
                    "hist_rate": m_hist['rate'],
                    "hist_cap": m_hist['avg_cap'],
                    
                    "garch_breaches": m_garch['breaches'],
                    "garch_rate": m_garch['rate'],
                    "garch_cap": m_garch['avg_cap'],
                    
                    "lstm_breaches": m_lstm['breaches'],
                    "lstm_rate": m_lstm['rate'],
                    "lstm_cap": m_lstm['avg_cap']
                })
                print(f"      [OK] Hist Rate:{m_hist['rate']}% | GARCH:{m_garch['rate']}% | LSTM:{m_lstm['rate']}%")

            except Exception as e:
                print(f"      [ERROR] {ticker}: {e}")

        self.save_results(results)

    def save_results(self, data):
        df_out = pd.DataFrame(data)
        table_id = f"{PROJECT_ID}.{DATASET_ID}.{OUTPUT_TABLE}"
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        self.client.load_table_from_dataframe(df_out, table_id, job_config=job_config).result()
        print(f"✅ Validator Batch Complete.")

if __name__ == "__main__":
    BatchRiskEngine().run_batch()