import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class SentimentBacktester:
    """
    Backtesting Engine for sentiment-based strategies (NLP).
    Simulates Long/Short order execution based on 'Sentiment Score'.
    """
    def __init__(self, initial_capital=100_000, commission_bps=5):
        self.initial_capital = initial_capital
        self.commission = commission_bps / 10000  # Basis points to decimal
        self.portfolio = pd.DataFrame()
        
    def run(self, df_prices, df_sentiment, threshold=0.1):
        """
        Executes the simulation.
        
        Args:
            df_prices (pd.DataFrame): Index=Date, Col='Close' (or 'Adj Close')
            df_sentiment (pd.DataFrame): Index=Date, Col='sentiment_score'
            threshold (float): Threshold to activate signal (e.g., > 0.1 is Buy)
        """
        print(f"⚡ Starting Alpha Backtest (Threshold: +/- {threshold})...")
        
        # 1. Align Data (Merge by Date)
        # We assume trading at close based on average sentiment of that day
        # (Or next day if using t-1 sentiment)
        data = df_prices[['Close']].copy()
        data.columns = ['Close']
        data = data.join(df_sentiment['sentiment_score'], how='inner').sort_index()
        # print(data.head())
        
        if data.empty:
            print("⚠️ Error: No date intersection between Prices and News.")
            return None
            
        # 2. Calculate Asset Returns
        data['market_return'] = data['Close'].pct_change()
        
        # 3. Generate Signals (Strategy)
        # Signal = 1 (Long), -1 (Short), 0 (Neutral)
        conditions = [
            (data['sentiment_score'] > threshold),  # Bullish News
            (data['sentiment_score'] < -threshold)  # Bearish News
        ]
        choices = [1, -1]
        # data['position'] = np.select(conditions, choices, default=0)
        data['position'] = 1

        # IMPORTANT: Signal Lag
        # We trade the NEXT day to avoid 'Look-ahead Bias'
        data['position'] = data['position'].shift(1)
        
        # 4. Calculate Strategy Returns
        # Strategy Return = Position * Market Return - Costs
        data['strategy_gross'] = data['position'] * data['market_return']
        
        # Transaction costs (applied when position changes)
        data['trades'] = data['position'].diff().abs()
        data['costs'] = data['trades'] * self.commission
        
        data['strategy_net'] = data['strategy_gross'] - data['costs']
        
        # 5. Equity Curve
        data['equity_market'] = self.initial_capital * (1 + data['market_return']).cumprod()
        data['equity_strategy'] = self.initial_capital * (1 + data['strategy_net']).cumprod()
        print(data.transpose())

        self.results = data.dropna()
        return self._calculate_metrics()

    def _calculate_metrics(self):
        """Calculates financial KPIs."""
        df = self.results
        print(df.head())
        total_return = (df['equity_strategy'].iloc[-1] / self.initial_capital) - 1
        market_return = (df['equity_market'].iloc[-1] / self.initial_capital) - 1
        
        # Sharpe Ratio (Annualized)
        daily_sharpe = df['strategy_net'].mean() / (df['strategy_net'].std() + 1e-9)
        sharpe_ratio = daily_sharpe * np.sqrt(252)
        
        # Max Drawdown
        peak = df['equity_strategy'].cummax()
        drawdown = (df['equity_strategy'] - peak) / peak
        max_drawdown = drawdown.min()
        
        metrics = {
            "Total Return": f"{total_return:.2%}",
            "Market Benchmark": f"{market_return:.2%}",
            "Alpha (Excess)": f"{total_return - market_return:.2%}",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}",
            "Max Drawdown": f"{max_drawdown:.2%}"
        }
        
        return metrics

    def plot_results(self, ticker_name="Asset"):
        """Performance Visualization."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.results.index, self.results['equity_strategy'], label='NLP Strategy (Alpha)', color='purple', linewidth=2)
        plt.plot(self.results.index, self.results['equity_market'], label='Buy & Hold (Benchmark)', color='gray', linestyle='--', alpha=0.7)
        
        plt.title(f"Alpha Strategy Backtest: {ticker_name}")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()