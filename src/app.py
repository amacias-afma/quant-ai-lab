# src/app.py
import os
import json
from flask import Flask, jsonify, request, render_template
from google.cloud import bigquery

app = Flask(__name__, template_folder='templates')
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "quant-ai-lab")
client = bigquery.Client(project=PROJECT_ID)

# --- RUTAS DE NAVEGACIÓN ---

@app.route('/')
def home():
    """Página de inicio (Portafolio de Proyectos)."""
    return render_template('index.html')

@app.route('/projects/volatility')
def project_volatility():
    """El Dashboard del Risk Engine (lo que antes era el index)."""
    return render_template('project_volatility.html')

# --- APIS DEL PROYECTO VOLATILIDAD (Sin cambios) ---

@app.route('/research/data', methods=['GET'])
def get_research_data():
    query = f"""
        SELECT 
            ticker, scenario, 
            hist_breach, garch_breach, lstm_breach,
            garch_capital AS capital_garch, 
            lstm_capital AS capital_lstm, 
            capital_savings
        FROM `{PROJECT_ID}.market_data.research_distribution`
    """
    try:
        df = client.query(query).to_dataframe()
        return jsonify(df.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/research/timeseries', methods=['GET'])
def get_research_timeseries():
    ticker = request.args.get('ticker')
    scenario = request.args.get('scenario')
    if not ticker or not scenario: return jsonify({"error": "Missing params"}), 400

    query = f"""
        SELECT timeseries_json
        FROM `{PROJECT_ID}.market_data.research_distribution`
        WHERE ticker = '{ticker}' AND scenario = '{scenario}'
        LIMIT 1
    """
    try:
        df = client.query(query).to_dataframe()
        if df.empty: return jsonify({"error": "Not found"}), 404
        return jsonify(json.loads(df.iloc[0]['timeseries_json']))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def get_forecast():
    ticker = request.json.get('ticker', 'SPY')
    query = f"SELECT * FROM `{PROJECT_ID}.market_data.dashboard_forecasts` WHERE ticker = '{ticker}' LIMIT 1"
    
    try:
        query_job = client.query(query)
        rows = list(query_job)
        if not rows: return jsonify({"error": "No data"}), 404
        row = rows[0]
        
        return jsonify({
            "status": "success",
            "ticker": row.ticker,
            "asset_name": row.asset_name,
            "metrics": {
                "hist_breaches": row.hist_breaches,
                "hist_rate": row.hist_rate,
                "hist_cap": row.hist_cap,
                
                "garch_breaches": row.garch_breaches,
                "garch_rate": row.garch_rate,
                "garch_cap": row.garch_cap,
                
                "lstm_breaches": row.lstm_breaches,
                "lstm_rate": row.lstm_rate,
                "lstm_cap": row.lstm_cap
            },
            "plot_data": {
                "dates": json.loads(row.dates_json),
                "pnl": json.loads(row.returns_json),
                "hist_var": json.loads(row.hist_var_json),
                "garch_var": json.loads(row.garch_var_json),
                "lstm_var": json.loads(row.lstm_var_json)
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))