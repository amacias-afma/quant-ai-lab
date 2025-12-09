# src/app.py
import os
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "Quantitative Volatility Engine",
        "version": "1.0.0"
    })

if __name__ == "__main__":
    # Local testing
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))