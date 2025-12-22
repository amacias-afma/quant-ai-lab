# src/models/batch_run.py
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
VAR_CONFIDENCE = 2.33 
HISTORICAL_QUANTILE = 0.01 

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

    def calculate_stress_metrics(self, df, start_date, end_date):
        """
        Calculates Breach Rate for a specific crisis period (e.g., 2020).
        """
        # 1. Slice Data for the Period
        mask = (df.index >= start_date) & (df.index <= end_date)
        df_period = df.loc[mask]
        
        if df_period.empty: return 0.0, 0.0
        
        # 2. Historical VaR (252d Rolling)
        # Note: We need the PREVIOUS 252 days to calculate returns for start_date
        # So we take the full series calculation and then slice.
        
        # Calculate Rolling 1% Quantile over the FULL history first
        # We assume df has the full history here
        full_rolling_var = df['log_ret'].rolling(252).quantile(HISTORICAL_QUANTILE).abs() * PORTFOLIO_SIZE
        
        # Now Slice the VaR and the Returns
        period_var = full_rolling_var.loc[mask]
        period_returns = df_period['log_ret'] * PORTFOLIO_SIZE
        period_losses = -period_returns
        
        # 3. Calculate Breaches
        breaches = period_losses > period_var
        breach_count = breaches.sum()
        breach_rate = (breach_count / len(df_period)) * 100
        
        return breach_rate

    def calculate_recent_metrics(self, returns, capital_series):
        """Standard metrics for the last 100 days (Recent Performance)"""
        n = len(returns)
        cap = np.array(capital_series[-n:])
        ret = np.array(returns[-n:]) * PORTFOLIO_SIZE
        losses = -ret
        breaches = losses > cap
        return cap.tolist(), int(np.sum(breaches)), (np.sum(breaches)/n)*100, float(np.mean(cap))

    def run_batch(self):
        print(f"🚀 Starting Validator Batch Run...")
        results = []

        for ticker, name in TICKER_MAP.items():
            print(f"   ... Processing {name}")
            try:
                df = self.load_ticker_data(ticker)
                if len(df) < 360: continue 

                # --- 1. STRESS TEST (2020 Crisis) ---
                # We prove the Historical Model fails here
                stress_rate_2020 = self.calculate_stress_metrics(df, '2020-01-01', '2020-12-31')

                # --- 2. RECENT MODELING (Last 100 Days) ---
                # GARCH
                garch = GarchBaseline(ticker)
                garch.train()
                garch_sigma = garch.model_fit.conditional_volatility / 100
                garch_capital_full = garch_sigma * VAR_CONFIDENCE * PORTFOLIO_SIZE
                
                # LSTM
                lstm = DeepVolEngine(ticker, seq_len=60)
                lstm.train(epochs=15)
                pred_vol, _ = lstm.predict(df) 
                lstm_sigma_daily = (pred_vol / np.sqrt(252))
                lstm_capital_full = lstm_sigma_daily * VAR_CONFIDENCE * PORTFOLIO_SIZE

                # Historical (For the chart)
                hist_var_rolling = df['log_ret'].rolling(252).quantile(HISTORICAL_QUANTILE).abs() * PORTFOLIO_SIZE

                # --- 3. METRICS ---
                eval_window = 100
                returns_slice = df['log_ret'][-eval_window:].tolist()
                
                # Recent Performance
                m_hist = self.calculate_recent_metrics(returns_slice, hist_var_rolling)
                m_garch = self.calculate_recent_metrics(returns_slice, garch_capital_full)
                m_lstm = self.calculate_recent_metrics(returns_slice, lstm_capital_full)

                # --- 4. PACKAGING ---
                results.append({
                    "ticker": ticker,
                    "asset_name": name,
                    "last_updated": pd.Timestamp.now(),
                    "dates_json": json.dumps(df.index[-eval_window:].strftime('%Y-%m-%d').tolist()),
                    "returns_json": json.dumps([x * PORTFOLIO_SIZE for x in returns_slice]),
                    
                    # Stress Test Score
                    "stress_rate_2020": stress_rate_2020,
                    
                    # Series
                    "hist_var_json": json.dumps(m_hist[0]),
                    "garch_var_json": json.dumps(m_garch[0]),
                    "lstm_var_json": json.dumps(m_lstm[0]),
                    
                    # Recent Metrics
                    "hist_breaches": m_hist[1], "hist_rate": m_hist[2], "hist_cap": m_hist[3],
                    "garch_breaches": m_garch[1], "garch_rate": m_garch[2], "garch_cap": m_garch[3],
                    "lstm_breaches": m_lstm[1], "lstm_rate": m_lstm[2], "lstm_cap": m_lstm[3]
                })
                print(f"      [OK] 2020 Fail Rate:{stress_rate_2020:.2f}% | Recent GARCH:{m_garch[2]}%")

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