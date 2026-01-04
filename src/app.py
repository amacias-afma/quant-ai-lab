# src/app.py
import os
import json
from flask import Flask, jsonify, request, render_template
from google.cloud import bigquery


from data.text import NewsIngestor
from models.alpha_rag.benchmark import VaderBenchmark
from models.alpha_rag.rag_engine import FinancialRAG

rag_system = FinancialRAG()

app = Flask(__name__, template_folder='templates')
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "quant-ai-lab")
client = bigquery.Client(project=PROJECT_ID)

# --- NAVIGATION ROUTES ---

@app.route('/')
def home():
    """Home Page (Project Portfolio)."""
    return render_template('index.html')

@app.route('/projects/volatility')
def project_volatility():
    """Risk Engine Dashboard (formerly the index)."""
    return render_template('project_volatility.html')

@app.route('/alpha/results', methods=['GET'])
def get_alpha_results():
    """Endpoint de lectura rápida para el Dashboard de Alpha."""
    ticker = request.args.get('ticker') # Opcional: filtrar por ticker
    
    base_query = f"""
        SELECT 
            ticker, total_return, alpha, sharpe, drawdown, chart_json 
        FROM `{PROJECT_ID}.market_data.alpha_results`
    """
    
    if ticker:
        base_query += f" WHERE ticker = '{ticker}'"
    
    try:
        df = client.query(base_query).to_dataframe()
        
        if df.empty:
            return jsonify({"status": "empty", "message": "Run study_alpha.py first"})
            
        # Convertir a lista de dicts (y parsear el JSON interno de la gráfica)
        results = []
        for _, row in df.iterrows():
            item = row.to_dict()
            item['chart_data'] = json.loads(row['chart_json']) # Deserializar
            del item['chart_json'] # Limpiar para no enviar doble data
            results.append(item)
            
        return jsonify({"status": "success", "data": results})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
@app.route('/projects/alpha-rag')
def project_alpha():
    """Project 2: Alpha Signals (NLP)."""
    return render_template('project_alpha.html')

# --- VOLATILITY PROJECT APIS (No Info Changes) ---

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

@app.route('/alpha/analyze', methods=['POST'])
def analyze_sentiment():
    """
    1. Download recent news.
    2. Execute Benchmark (VADER).
    3. Execute RAG (Embeddings + LLM).
    4. Return comparison.
    """
    data = request.json
    ticker = data.get('ticker', 'SPY')
    
    try:
        # A. Ingestion
        ingestor = NewsIngestor(ticker)
        df_news = ingestor.fetch_recent_news()
        
        if df_news.empty:
            return jsonify({"status": "error", "message": "No news found for this ticker."})

        # B. Benchmark (VADER)
        vader = VaderBenchmark()
        df_scored = vader.analyze_dataframe(df_news)
        vader_score = df_scored['sentiment_score'].mean()
        
        # C. RAG (Advanced)
        # Load news into RAG memory
        rag_system.ingest_data(df_news)
        # Request qualitative insight
        rag_insight = rag_system.get_trading_signal(ticker)
        
        # D. Response
        return jsonify({
            "status": "success",
            "ticker": ticker,
            "news_count": len(df_news),
            "benchmark_score": vader_score, # Number between -1 and 1
            "rag_insight": rag_insight,     # Explanatory text from LLM
            "top_news": df_scored.nlargest(3, 'sentiment_score')[['title', 'link', 'sentiment_score']].to_dict(orient='records')
        })
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        return jsonify({"error": str(e)}), 500
        
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))