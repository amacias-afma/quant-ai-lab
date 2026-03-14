"""
Visualization modules for Quantitative Analysis.

This module contains reusable plotting functions for Exploratory Data Analysis (EDA),
model evaluation (PIT histograms, CRPS rolling plots), and risk metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_price_analysis(prices: pd.Series, returns: pd.Series, ticker_name: str) -> None:
    """
    Generates a 2x2 dashboard of stylized financial facts for a given asset.
    Includes Price History, Daily Returns, Return Distribution, and Rolling Volatility.
    
    Args:
        prices: pd.Series of asset prices with a DatetimeIndex.
        returns: pd.Series of asset daily returns with a DatetimeIndex.
        ticker_name: String name of the asset (e.g., 'BTC', 'SPY').
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"{ticker_name} — Price Analysis", fontsize=15, fontweight="bold")

    # 1. Price history
    ax = axes[0, 0]
    ax.plot(prices.index, prices, color="#F7931A", linewidth=1.2)
    ax.set_title("Close Price History")
    ax.set_ylabel("USD")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # 2. Daily returns
    ax = axes[0, 1]
    ax.bar(returns.index, returns, color=np.where(returns >= 0, "#2ecc71", "#e74c3c"), width=1, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Daily Returns")
    ax.set_ylabel("Return")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # 3. Return distribution
    ax = axes[1, 0]
    ax.hist(returns, bins=80, color="#3498db", edgecolor="none", alpha=0.8, density=True)
    ax.axvline(returns.mean(), color="red", linestyle="--", linewidth=1.2, label=f"Mean {returns.mean():.4f}")
    ax.axvline(returns.quantile(0.05), color="orange", linestyle="--", linewidth=1.2, label="5th pct")
    ax.set_title("Return Distribution")
    ax.set_xlabel("Daily Return")
    ax.legend(fontsize=8)

    # 4. Rolling 30-day volatility
    rolling_vol = returns.rolling(30).std() * np.sqrt(252)
    ax = axes[1, 1]
    ax.plot(rolling_vol.index, rolling_vol, color="#9b59b6", linewidth=1.2)
    ax.set_title("Rolling 30-day Annualized Volatility")
    ax.set_ylabel("Volatility")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.show()


