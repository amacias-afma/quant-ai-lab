# src/app.py
import os
import json
import numpy as np
from flask import Flask, jsonify, request

# Import our custom engines
from src.models.baseline import GarchBaseline
from src.models.lstm import DeepVolEngine
# Note: Ensure you have empty __init__.py files in your directories to make them packages

app = Flask(__name__)
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "quant-ai-lab")

def convert_numpy(obj):
    """Helper to convert Numpy types to Python native (for JSON serialization)."""
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

@app.route('/')
def health_check():
    return jsonify({
        "status": "ready",
        "service": "Hybrid Volatility Engine (GARCH + LSTM)",
        "version": "2.0.0"
    })

@app.route('/predict', methods=['POST', 'GET'])
def predict_risk():
    """
    Main Endpoint: Triggers the analysis pipeline.
    """
    # 1. Parse Request
    ticker = request.args.get('ticker', 'SPY')
    print(f"--- Starting Analysis for {ticker} ---")
    
    try:
        # ---------------------------------------------------------
        # A. GARCH MODEL (The Benchmark)
        # ---------------------------------------------------------
        garch_engine = GarchBaseline(project_id=PROJECT_ID, ticker=ticker)
        df_garch = garch_engine.load_data()
        
        # Train GARCH
        garch_engine.train(df_garch)
        
        # Predict (Next Day Volatility)
        garch_vol = garch_engine.forecast_volatility()
        
        # ---------------------------------------------------------
        # B. LSTM MODEL (The Challenger)
        # ---------------------------------------------------------
        # For the demo, we use fewer epochs to ensure the HTTP request doesn't timeout
        lstm_engine = DeepVolEngine(project_id=PROJECT_ID, ticker=ticker, seq_len=60)
        df_lstm = lstm_engine.load_data()
        
        # Train LSTM (Fast mode for web demo)
        lstm_engine.train(df_lstm, epochs=10, batch_size=32)
        
        # Predict (Get the last predicted volatility)
        pred_vol_series, _ = lstm_engine.predict(df_lstm)
        lstm_vol_next_day = pred_vol_series[-1]
        
        # ---------------------------------------------------------
        # C. CONSTRUCT RESPONSE
        # ---------------------------------------------------------
        response = {
            "ticker": ticker,
            "forecasts": {
                "GARCH_vol_annualized": garch_vol,
                "LSTM_vol_annualized": lstm_vol_next_day
            },
            "model_comparison": {
                "status": "success",
                "note": "LSTM uses QLIKE loss (Risk-Averse), GARCH uses Normal Likelihood."
            }
        }
        
        return json.dumps(response, default=convert_numpy), 200

    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({"error": str(e), "trace": "Check Cloud Run logs for details."}), 500

if __name__ == "__main__":
    # Local testing
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))