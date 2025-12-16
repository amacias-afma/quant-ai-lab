# src/app.py
import os
import json
from flask import Flask, jsonify, request, render_template
from google.cloud import bigquery

app = Flask(__name__, template_folder='templates')
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "quant-ai-lab")
client = bigquery.Client(project=PROJECT_ID)

# asset_dictionary = {}

@app.route('/')
def home():
    return render_template('index.html')

def get_asset_dictionary():
    """Returns list of {code, name} for dropdown."""
    query = f"""
        SELECT DISTINCT ticker, asset_name 
        FROM `{PROJECT_ID}.market_data.historical_prices`
        ORDER BY ticker
    """
    try:
        query_job = client.query(query)
        # Return object for cleaner UI
        # data = [{"code": row.ticker, "name": row.asset_name} for row in query_job]
        asset_dictionary = {row.ticker: row.asset_name for row in query_job}
        return asset_dictionary
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/tickers', methods=['GET'])
def get_tickers():
    """Returns list of {code, name} for dropdown."""
    # query = f"""
    #     SELECT DISTINCT ticker, asset_name 
    #     FROM `{PROJECT_ID}.market_data.historical_prices`
    #     ORDER BY ticker
    # """
    try:
        asset_dictionary = get_asset_dictionary()

        # query_job = client.query(query)
        # Return object for cleaner 
        data = [{"code": key, "name": item} for key, item in asset_dictionary.items()]
        return jsonify(data)
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



# @app.route('/predict', methods=['POST'])
# def get_forecast():
#     ticker = request.json.get('ticker', 'SPY')
#     asset_dictionary = get_asset_dictionary()
#     asset_name = asset_dictionary.get(ticker, 'Unknown')
#     query = f"SELECT * FROM `{PROJECT_ID}.market_data.dashboard_forecasts` WHERE ticker = '{ticker}' LIMIT 1"
    
#     # try:
#     query_job = client.query(query)
#     rows = list(query_job)
#     if not rows: return jsonify({"error": "No data"}), 404
#     row = rows[0]
#     # print(row)
#     # print('-'*20)
#     # print(row.lstm_var_json)
#     # print('-'*20)

#     return jsonify({
#         "status": "success",
#         "ticker": row.ticker,
#         "asset_name": asset_name, # <--- Pass name to UI
#         "metrics": {
#             "garch_breaches": row.garch_breaches,
#             "garch_avg_cap": 0,
#             "lstm_breaches": row.lstm_breaches,
#             "lstm_avg_cap": 0,
#             "real_avg_cap": 0

#             # "garch_breaches": row.garch_breaches,
#             # "garch_avg_cap": row.garch_avg_cap,
#             # "lstm_breaches": row.lstm_breaches,
#             # "lstm_avg_cap": row.lstm_avg_cap,
#             # "real_avg_cap": row.real_avg_cap
#         },
#         "plot_data": {
#             "dates": json.loads(row.dates_json),
#             "pnl": json.loads(row.returns_json),
#             "garch_var": json.loads(row.garch_var_json),
#             "lstm_var": json.loads(row.lstm_var_json),
#             "real_var": json.loads(row.real_var_json) # <--- Pass Realized VaR
#         }
#     })
#     # except Exception as e:
#     #     return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))