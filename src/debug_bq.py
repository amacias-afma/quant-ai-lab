import os
import sys

# Redirect output to file
log_file = open("debug_bq.log", "w")
sys.stdout = log_file
sys.stderr = log_file

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "quant-ai-lab")
print(f"Using Project ID: {PROJECT_ID}")

try:
    from google.cloud import bigquery
    client = bigquery.Client(project=PROJECT_ID)
    
    # FIXED Query
    query = f"""
        SELECT 
            ticker, scenario, 
            hist_breach, garch_breach, lstm_breach,
            garch_capital AS capital_garch, 
            lstm_capital AS capital_lstm, 
            capital_savings
        FROM `{PROJECT_ID}.market_data.research_distribution`
    """
    
    print("Running FIXED query...")
    df = client.query(query).to_dataframe()
    print("Query success!")
    print(df.head())
    
except Exception as e:
    print(f"Query FAILED: {e}")

log_file.close()
