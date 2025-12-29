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
    # distribution = get_distribution_data()
    # print(distribution)
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

# src/app.py

@app.route('/research/timeseries', methods=['GET'])
def get_research_timeseries():
    """Fetches the detailed time-series for a specific Ticker + Scenario."""
    ticker = request.args.get('ticker')
    scenario = request.args.get('scenario')
    
    if not ticker or not scenario:
        return jsonify({"error": "Missing parameters"}), 400

    query = f"""
        SELECT timeseries_json
        FROM `{PROJECT_ID}.market_data.research_distribution`
        WHERE ticker = '{ticker}' AND scenario = '{scenario}'
        LIMIT 1
    """
    
    try:
        df = client.query(query).to_dataframe()
        if df.empty: return jsonify({"error": "Not found"}), 404
        
        # The data is stored as a JSON string in BigQuery, so we parse it back
        raw_json = df.iloc[0]['timeseries_json']
        return jsonify(json.loads(raw_json))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/tickers', methods=['GET'])
def get_market_overview():
    """Returns summary for the Market Monitor table."""
    query = f"""
        SELECT 
            ticker, 
            asset_name, 
            stress_rate_2020,  -- NEW FIELD
            hist_rate,         -- Recent Historical Rate
            garch_rate,        -- Recent GARCH Rate
            lstm_rate          -- Recent LSTM Rate
        FROM `{PROJECT_ID}.market_data.dashboard_forecasts`
        ORDER BY ticker
    """
    try:
        query_job = client.query(query)
        data = []
        for row in query_job:
            data.append({
                "code": row.ticker,
                "name": row.asset_name,
                "stress_2020": row.stress_rate_2020,
                "hist_recent": row.hist_rate,
                "garch_recent": row.garch_rate,
                "lstm_recent": row.lstm_rate
            })
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def get_distribution_data():
    """Returns granular breach rates for the box/scatter plot."""
    
    query = f"""
        SELECT * FROM `{PROJECT_ID}.market_data.research_distribution`
    """
    try:
        df = client.query(query).to_dataframe()
        return df
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/research/distribution', methods=['GET'])
def get_distribution():
    """Returns granular breach rates for the box/scatter plot."""
    query = f"""
        SELECT * FROM `{PROJECT_ID}.market_data.research_distribution`
    """
    try:
        df = client.query(query).to_dataframe()
        df.columns = df.columns.str.lower()
        # print("df "*5)
        # print(df)
        # print("- "*10)

        # Format for Chart.js Scatter
        # We map models to X-coordinates: Hist=1, GARCH=2, LSTM=3
        datasets = []
        
        models = [
            ('hist_breach', 'Historical (Benchmark)', '#6c757d'),
            ('garch_breach', 'GARCH (Parametric)', '#dc3545'),
            ('lstm_breach', 'Deep LSTM (AI)', '#198754')
        ]
        
        for col, label, color in models:
            points = []
            print(col, col.lower())
            for val in df[col.lower()].values:
                # Add jitter to X for visibility (Strip Plot effect)
                print(val)
                jitter = (hash(str(val)) % 100) / 500 - 0.1 
                x_base = models.index((col.lower(), label, color)) + 1
                points.append({'x': x_base + jitter, 'y': val})
                
            datasets.append({
                'label': label,
                'data': points,
                'backgroundColor': color,
                'borderColor': color,
                'pointRadius': 4
            })
        
        # print("datasets"*5)
        # print(datasets) 
        return jsonify(datasets)
    except Exception as e:
        # print('error '*5)
        # print(e)
        return jsonify({"error": str(e)}), 500

# @app.route('/tickers', methods=['GET'])
# def get_tickers():
#     """Returns list of {code, name} for dropdown."""
#     # query = f"""
#     #     SELECT DISTINCT ticker, asset_name 
#     #     FROM `{PROJECT_ID}.market_data.historical_prices`
#     #     ORDER BY ticker
#     # """
#     try:
#         asset_dictionary = get_asset_dictionary()

#         # query_job = client.query(query)
#         # Return object for cleaner 
#         data = [{"code": key, "name": item} for key, item in asset_dictionary.items()]
#         return jsonify(data)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

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

# src/app.py (Add this endpoint)

@app.route('/research/data', methods=['GET'])
def get_research_data():
    """Returns ALL raw research results for client-side processing."""
    query = f"""
        SELECT 
            ticker, scenario, 
            hist_breach, garch_breach, lstm_breach,
            garch_capital, lstm_capital, capital_savings
        FROM `{PROJECT_ID}.market_data.research_distribution`
    """
    df = client.query(query).to_dataframe()
    # Convert to list of dicts
    # print(df.head())
    # data = df.to_dict(orient='records')
    # print(data)
    try:
        df = client.query(query).to_dataframe()
        # Convert to list of dicts
        data = df.to_dict(orient='records')
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/research/summary', methods=['GET'])
def get_research_summary():
    """Returns aggregated research conclusions."""
    # Fetch ALL results (No WHERE clause)
    query = f"""
        SELECT * FROM `{PROJECT_ID}.market_data.research_results`
    """
    try:
        df = client.query(query).to_dataframe()
        
        # 1. Simulation Count
        sim_count = len(df)
        
        # 2. Capital Efficiency % (Global)
        total_garch_cap = df['capital_garch'].sum()
        total_lstm_cap = df['capital_lstm'].sum()
        
        if total_garch_cap > 0:
            efficiency = ((total_garch_cap - total_lstm_cap) / total_garch_cap) * 100
        else:
            efficiency = 0.0

        return jsonify({
            "sim_count": sim_count,
            "lstm_win_rate": (df['lstm_breach'] < df['garch_breach']).mean() * 100,
            "avg_lstm_breach": df['lstm_breach'].mean(),
            "avg_garch_breach": df['garch_breach'].mean(),
            "capital_efficiency_pct": efficiency,
            # We just return the first 10 rows for the breakdown table to keep payload small
            "details": df.head(10).to_dict(orient='records') 
        })
    except Exception as e:
        print('error get_research_summary'*5)
        print(e)
        return jsonify({"error": str(e)}), 500

# @app.route('/research/summary', methods=['GET'])
# def get_research_summary():
#     """Returns the aggregated research conclusions."""
#     query = f"""
#         SELECT * FROM `{PROJECT_ID}.market_data.research_distribution`
#     """
#     df = client.query(query).to_dataframe()
#     print('df.head() '*5)

#     print(df.head())
#     # Calculate quantitative conclusions
#     avg_lstm_breach = df['lstm_breach'].mean()
#     avg_garch_breach = df['garch_breach'].mean()
#     total_savings = df['capital_savings'].sum()
    
#     return jsonify({
#         "avg_lstm_breach": avg_lstm_breach,
#         "avg_garch_breach": avg_garch_breach,
#         "lstm_win_rate": (df['lstm_breach'] < df['garch_breach']).mean() * 100,
#         "total_capital_impact": total_savings,
#         "details": df.to_dict(orient='records')
#     })

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