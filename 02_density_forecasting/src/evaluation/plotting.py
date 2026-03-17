"""
Visualization modules for Quantitative Analysis.

This module contains reusable plotting functions for Exploratory Data Analysis (EDA),
model evaluation (PIT histograms, CRPS rolling plots), and risk metrics.
"""

import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def compute_descriptive_stats(prices: pd.Series, returns: pd.Series) -> Styler:
    """
    Computes and returns a styled descriptive statistics table for price and returns.

    Args:
        prices:  pd.Series of asset Close prices with a DatetimeIndex.
        returns: pd.Series of asset daily returns with a DatetimeIndex.

    Returns:
        pandas Styler object — call display() on it in a Jupyter cell.
    """
    ann_vol = returns.std() * np.sqrt(252)
    sharpe  = (returns.mean() / returns.std()) * np.sqrt(252)
    max_dd  = (prices / prices.cummax() - 1).min()

    stats = pd.DataFrame({
        "Close Price (USD)": {
            "Start":       prices.iloc[0],
            "End":         prices.iloc[-1],
            "Min":         prices.min(),
            "Max":         prices.max(),
            "Mean":        prices.mean(),
            "Std Dev":     prices.std(),
            "Skewness":    np.nan,
            "Kurtosis":    np.nan,
            "Ann. Vol":    np.nan,
            "Sharpe (252)": np.nan,
            "Max Drawdown": np.nan,
        },
        "Daily Returns": {
            "Start":       returns.iloc[0],
            "End":         returns.iloc[-1],
            "Min":         returns.min(),
            "Max":         returns.max(),
            "Mean":        returns.mean(),
            "Std Dev":     returns.std(),
            "Skewness":    returns.skew(),
            "Kurtosis":    returns.kurt(),
            "Ann. Vol":    ann_vol,
            "Sharpe (252)": sharpe,
            "Max Drawdown": max_dd,
        },
    }).round(4)

    def _highlight_extremes(col):
        """Color positive good / negative bad cells for the Returns column."""
        if col.name != "Daily Returns":
            return [""] * len(col)
        styles = []
        for idx, val in col.items():
            if pd.isna(val):
                styles.append("")
            elif idx in ("Max", "Mean", "Sharpe (252)", "Ann. Vol") and val > 0:
                styles.append("background-color: #d4edda; color: #155724;")  # green
            elif idx in ("Min", "Max Drawdown", "Kurtosis") and val < 0:
                styles.append("background-color: #f8d7da; color: #721c24;")  # red
            else:
                styles.append("")
        return styles

    styled = (
        stats.style
        .apply(_highlight_extremes, axis=0)
        .format(na_rep="—", precision=4)
        .set_caption("📊 Descriptive Statistics")
        .set_table_styles([
            {"selector": "caption",
             "props": [("font-size", "14px"), ("font-weight", "bold"),
                       ("text-align", "left"), ("padding-bottom", "6px")]},
            {"selector": "th",
             "props": [("background-color", "#2c3e50"), ("color", "white"),
                       ("font-weight", "bold"), ("padding", "6px 12px")]},
            {"selector": "td",
             "props": [("padding", "5px 12px"), ("border-bottom", "1px solid #dee2e6")]},
            {"selector": "tr:hover td",
             "props": [("background-color", "#f0f4f8")]},
        ])
    )
    return styled


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


def plot_forecast_evaluation(df_ret, ticker_name='PERFECT WORLD'):
    """
    Plots the PIT Histogram and Rolling CRPS to visually evaluate forecast calibration.
    """
    plt.style.use('bmh')
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: The PIT Histogram (Should be Flat!)
    axes[0].hist(df_ret['PIT'], bins=20, edgecolor='black', alpha=0.7, color='mediumseagreen', density=True)
    axes[0].axhline(1, color='red', linestyle='--', linewidth=2, label='Theoretical Uniform (Perfect)')
    axes[0].set_title(f"PIT Histogram (Calibrated) for {ticker_name}")
    axes[0].set_xlabel("Cumulative Probability ($u_t$)")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    # Plot 2: Rolling CRPS (Should be stable and low)
    axes[1].plot(df_ret['CRPS'].rolling(20).mean(), color='darkorange')
    axes[1].axhline(df_ret['CRPS'].mean(), color='black', linestyle='--', label='Average CRPS')
    axes[1].set_title(f"Rolling CRPS for {ticker_name}")
    axes[1].set_xlabel("Time (Days)")
    axes[1].set_ylabel("CRPS Score")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_forecast_distribution(
    samples: np.ndarray,
    ticker_name: str = "",
    rf_daily: float = 0.0,
    periods_per_year: int = 252,
) -> None:
    """
    Simple visual summary of a forecasted return distribution.

    Left panel  — Histogram of the simulated returns with VaR 99% marked.
    Right panel — Clean table with 5 key risk/return metrics.

    Args:
        samples         : 1-D array of simulated daily return samples.
        ticker_name     : Optional label shown in the figure title.
        rf_daily        : Daily risk-free rate (default 0).
        periods_per_year: Trading days per year for annualisation (default 252).
    """
    s = np.asarray(samples, dtype=float)
    s = s[~np.isnan(s)]
    ann = np.sqrt(periods_per_year)

    # ── Compute the 5 key metrics ────────────────────────────────────────────
    exp_return   = np.mean(s)
    volatility   = np.std(s, ddof=1)
    var_99       = np.percentile(s, 1.0)          # 99 % VaR (1st percentile)
    cvar_99      = s[s <= var_99].mean() if np.any(s <= var_99) else var_99
    excess       = s - rf_daily
    sharpe       = (np.mean(excess) / np.std(s, ddof=1)) * ann if np.std(s) > 0 else np.nan

    metrics = {
        "Expected Return (daily)": f"{exp_return:+.5f}",
        "Volatility      (daily)": f"{volatility:.5f}",
        "VaR  99%        (daily)": f"{var_99:+.5f}",
        "CVaR 99%        (daily)": f"{cvar_99:+.5f}",
        "Sharpe Ratio    (ann.)":  f"{sharpe:.4f}" if not np.isnan(sharpe) else "—",
    }

    # ── Figure layout ────────────────────────────────────────────────────────
    plt.style.use("bmh")
    fig, (ax_hist, ax_table) = plt.subplots(
        1, 2, figsize=(12, 4),
        gridspec_kw={"width_ratios": [2, 1]},
    )
    title = f"Forecast Distribution — {ticker_name}" if ticker_name else "Forecast Distribution"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # ── Left: Histogram ──────────────────────────────────────────────────────
    ax_hist.hist(s, bins=50, density=True, color="#3498db", edgecolor="none", alpha=0.75)
    ax_hist.axvline(exp_return, color="gold",    linestyle="--", linewidth=1.5, label=f"E[r] = {exp_return:+.4f}")
    ax_hist.axvline(var_99,     color="#e74c3c", linestyle="--", linewidth=1.5, label=f"VaR 99% = {var_99:+.4f}")
    ax_hist.axvline(cvar_99,    color="#c0392b", linestyle=":",  linewidth=1.5, label=f"CVaR 99% = {cvar_99:+.4f}")
    ax_hist.set_xlabel("Daily Return")
    ax_hist.set_ylabel("Density")
    ax_hist.legend(fontsize=8)

    # ── Right: Metrics table ─────────────────────────────────────────────────
    ax_table.axis("off")
    rows  = list(metrics.keys())
    vals  = list(metrics.values())
    table = ax_table.table(
        cellText  =[[v] for v in vals],
        rowLabels =rows,
        colLabels =["Value"],
        cellLoc   ="center",
        rowLoc    ="left",
        loc       ="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)

    # Style header and row-label cells
    for (row, col), cell in table.get_celld().items():
        if row == 0 or col == -1:           # header row or row labels
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f8f9fa")
        cell.set_edgecolor("#dee2e6")

    plt.tight_layout()
    plt.show()
