"""
Visualization for Density Forecasting.

This module contains reusable plotting functions organized in three layers:

1. EDA (Exploratory Data Analysis):
   - compute_descriptive_stats  — Styled summary statistics table
   - plot_price_analysis        — 2×2 dashboard: price, returns, distribution, vol

2. Model Evaluation:
   - plot_forecast_evaluation   — PIT histogram + rolling CRPS
   - plot_rolling_ks            — Rolling K-S p-value with danger-zone shading

3. Risk Summary:
   - plot_forecast_distribution — Histogram + key risk/return metrics table
"""

import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ---------------------------------------------------------------------------
# 1. EDA
# ---------------------------------------------------------------------------

def compute_descriptive_stats(prices: pd.Series, returns: pd.Series) -> Styler:
    """
    Compute and return a styled descriptive statistics table.

    Covers both the price series and the returns series, highlighting
    positive (green) and negative (red) values in the Returns column.

    Args:
        prices:  pd.Series of asset close prices with a DatetimeIndex.
        returns: pd.Series of asset daily returns with a DatetimeIndex.

    Returns:
        pandas Styler object — call display() on it in a Jupyter cell.
    """
    ann_vol = returns.std() * np.sqrt(252)
    sharpe  = (returns.mean() / returns.std()) * np.sqrt(252)
    max_dd  = (prices / prices.cummax() - 1).min()

    stats = pd.DataFrame({
        "Close Price (USD)": {
            "Start":        prices.iloc[0],
            "End":          prices.iloc[-1],
            "Min":          prices.min(),
            "Max":          prices.max(),
            "Mean":         prices.mean(),
            "Std Dev":      prices.std(),
            "Skewness":     np.nan,
            "Kurtosis":     np.nan,
            "Ann. Vol":     np.nan,
            "Sharpe (252)": np.nan,
            "Max Drawdown": np.nan,
        },
        "Daily Returns": {
            "Start":        returns.iloc[0],
            "End":          returns.iloc[-1],
            "Min":          returns.min(),
            "Max":          returns.max(),
            "Mean":         returns.mean(),
            "Std Dev":      returns.std(),
            "Skewness":     returns.skew(),
            "Kurtosis":     returns.kurt(),
            "Ann. Vol":     ann_vol,
            "Sharpe (252)": sharpe,
            "Max Drawdown": max_dd,
        },
    }).round(4)

    def _highlight_extremes(col):
        """Color positive-good / negative-bad cells for the Returns column."""
        if col.name != "Daily Returns":
            return [""] * len(col)
        styles = []
        for idx, val in col.items():
            if pd.isna(val):
                styles.append("")
            elif idx in ("Max", "Mean", "Sharpe (252)", "Ann. Vol") and val > 0:
                styles.append("background-color: #d4edda; color: #155724;")   # green
            elif idx in ("Min", "Max Drawdown", "Kurtosis") and val < 0:
                styles.append("background-color: #f8d7da; color: #721c24;")   # red
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
                       ("text-align", "left"),  ("padding-bottom", "6px")]},
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
    2×2 dashboard of stylized financial facts for a given asset.

    Panels:
        [0,0] Price history (line)
        [0,1] Daily returns (bar chart, green/red)
        [1,0] Return distribution (histogram + mean and 5th-pct lines)
        [1,1] Rolling 30-day annualized volatility

    Args:
        prices:      pd.Series of asset prices with a DatetimeIndex.
        returns:     pd.Series of asset daily returns with a DatetimeIndex.
        ticker_name: Asset label shown in the title (e.g. 'BTC', 'SPY').
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"{ticker_name} — Price Analysis", fontsize=15, fontweight="bold")

    # [0,0] Price history
    ax = axes[0, 0]
    ax.plot(prices.index, prices, color="#F7931A", linewidth=1.2)
    ax.set_title("Close Price History")
    ax.set_ylabel("USD")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # [0,1] Daily returns (green = positive, red = negative)
    ax = axes[0, 1]
    ax.bar(returns.index, returns,
           color=np.where(returns >= 0, "#2ecc71", "#e74c3c"), width=1, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Daily Returns")
    ax.set_ylabel("Return")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # [1,0] Return distribution
    ax = axes[1, 0]
    ax.hist(returns, bins=80, color="#3498db", edgecolor="none", alpha=0.8, density=True)
    ax.axvline(returns.mean(),        color="red",    linestyle="--", linewidth=1.2,
               label=f"Mean {returns.mean():.4f}")
    ax.axvline(returns.quantile(0.05), color="orange", linestyle="--", linewidth=1.2,
               label="5th pct")
    ax.set_title("Return Distribution")
    ax.set_xlabel("Daily Return")
    ax.legend(fontsize=8)

    # [1,1] Rolling 30-day annualized volatility
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


# ---------------------------------------------------------------------------
# 2. Model Evaluation
# ---------------------------------------------------------------------------

def plot_forecast_evaluation(df_ret: pd.DataFrame, ticker_name: str = "PERFECT WORLD") -> None:
    """
    PIT histogram and rolling CRPS — the two core calibration diagnostics.

    Left panel:  PIT histogram.  A perfectly calibrated model produces a
                 flat (uniform) histogram.  Departures reveal systematic bias:
                 U-shaped ↔ overdispersion, hump-shaped ↔ underdispersion.

    Right panel: Rolling 20-day mean CRPS.  Should be stable and low.
                 Spikes indicate periods where the forecast was off.

    Args:
        df_ret:      DataFrame returned by `evaluate_forecasts`.
        ticker_name: Label shown in the plot titles.
    """
    plt.style.use('bmh')
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Left: PIT histogram
    axes[0].hist(df_ret['PIT'], bins=20, edgecolor='black',
                 alpha=0.7, color='mediumseagreen', density=True)
    axes[0].axhline(1, color='red', linestyle='--', linewidth=2,
                    label='Theoretical Uniform (Perfect)')
    axes[0].set_title(f"PIT Histogram (Calibrated) for {ticker_name}")
    axes[0].set_xlabel("Cumulative Probability ($u_t$)")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # Right: Rolling CRPS
    axes[1].plot(df_ret['CRPS'].rolling(20).mean(), color='darkorange')
    axes[1].axhline(df_ret['CRPS'].mean(), color='black', linestyle='--',
                    label='Average CRPS')
    axes[1].set_title(f"Rolling CRPS for {ticker_name}")
    axes[1].set_xlabel("Time (Days)")
    axes[1].set_ylabel("CRPS Score")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_rolling_ks(
    ax, 
    ticker_name, 
    volatile_periods, 
    pvals, 
    i=0
):
    # Draw the Volatility Regimes (Shaded Background - Gray so it doesn't clash with the red failure zone)
    for j, (event_name, (start_date, end_date)) in enumerate(volatile_periods.items()):
        label = event_name if i == 0 else ""
        ax.axvspan(pd.to_datetime(start_date), pd.to_datetime(end_date), 
                    color='gray', alpha=0.15, label=label)

    # Plot the P-Values
    for pvals_dict in pvals:
        ax.plot(pvals_dict['values'], 
            label=pvals_dict['name'], 
            color=pvals_dict['color'], 
            alpha=0.6, 
            linewidth=1.5)

        # ax.plot(pvals_vix['values'], label='VIX-Scaled Student-t (Forward)', color='blue', linewidth=2.0)

    # The Critical 0.05 Threshold
    ax.axhline(0.05, color='black', linestyle='--', linewidth=2, label='0.05 Significance Threshold' if i == 0 else "")

    # Shade the Failure Zone (p < 0.05)
    ax.axhspan(0.0, 0.05, color='red', alpha=0.2, label='Model Failure Zone (p < 0.05)' if i == 0 else "")

    ax.set_title(f"{ticker_name} Rolling Calibration Test", fontsize=14, fontweight='bold')
    ax.set_ylabel("K-S Test P-Value")
    ax.set_ylim(0.0, 0.1) 



# def plot_rolling_ks(
#     p_values: pd.Series,
#     ticker_name: str = "Model",
#     alpha: float = 0.05,
# ) -> None:
#     """
#     Rolling K-S p-value over time — dynamic calibration alarm chart.

#     Highlights the "danger zone" (p-value < α) where the model has locally
#     lost calibration and its forecasted distribution departs from reality.

#     Args:
#         p_values:    pd.Series of rolling K-S p-values (from `rolling_ks_test`).
#         ticker_name: Asset/model label shown in the title.
#         alpha:       Rejection threshold drawn as a red dashed line.
#     """
#     plt.style.use('bmh')
#     plt.figure(figsize=(14, 4))

#     plt.plot(p_values.index, p_values, color='indigo', linewidth=1.5,
#              label='Rolling K-S p-value')
#     plt.axhline(alpha, color='red', linestyle='--', linewidth=2,
#                 label=f'Rejection Threshold (α={alpha})')
#     plt.fill_between(p_values.index, 0, alpha, color='red', alpha=0.2,
#                      label='Danger Zone')

#     plt.title(f"Dynamic Calibration Alarm: Rolling K-S Test for {ticker_name}")
#     plt.ylabel("p-value")
#     plt.xlabel("Time")
#     plt.legend(loc="upper left")
#     plt.tight_layout()
#     ax.set_ylim(0.0, ylim) 

#     plt.show()


# ---------------------------------------------------------------------------
# 3. Risk Summary
# ---------------------------------------------------------------------------

def plot_forecast_distribution(
    samples: np.ndarray,
    ticker_name: str = "",
    rf_daily: float = 0.0,
    periods_per_year: int = 252,
) -> None:
    """
    Visual summary of a forecasted return distribution.

    Left panel:  Histogram of simulated daily returns with vertical lines
                 for the expected return, VaR 99%, and CVaR 99%.
    Right panel: Clean table of five key risk/return metrics.

    Args:
        samples:          1-D array of simulated daily return samples.
        ticker_name:      Optional label shown in the figure title.
        rf_daily:         Daily risk-free rate (default 0).
        periods_per_year: Trading days per year for annualisation (default 252).
    """
    s   = np.asarray(samples, dtype=float)
    s   = s[~np.isnan(s)]
    ann = np.sqrt(periods_per_year)

    # Compute the 5 key metrics
    exp_return = np.mean(s)
    volatility = np.std(s, ddof=1)
    var_99     = np.percentile(s, 1.0)
    cvar_99    = s[s <= var_99].mean() if np.any(s <= var_99) else var_99
    excess     = s - rf_daily
    sharpe     = (np.mean(excess) / np.std(s, ddof=1)) * ann if np.std(s) > 0 else np.nan

    metrics = {
        "Expected Return (daily)": f"{exp_return:+.5f}",
        "Volatility      (daily)": f"{volatility:.5f}",
        "VaR  99%        (daily)": f"{var_99:+.5f}",
        "CVaR 99%        (daily)": f"{cvar_99:+.5f}",
        "Sharpe Ratio    (ann.)":  f"{sharpe:.4f}" if not np.isnan(sharpe) else "—",
    }

    # Figure layout: histogram (2/3) + table (1/3)
    plt.style.use("bmh")
    fig, (ax_hist, ax_table) = plt.subplots(
        1, 2, figsize=(12, 4),
        gridspec_kw={"width_ratios": [2, 1]},
    )
    title = f"Forecast Distribution — {ticker_name}" if ticker_name else "Forecast Distribution"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Left: Histogram
    ax_hist.hist(s, bins=50, density=True, color="#3498db", edgecolor="none", alpha=0.75)
    ax_hist.axvline(exp_return, color="gold",    linestyle="--", linewidth=1.5,
                    label=f"E[r] = {exp_return:+.4f}")
    ax_hist.axvline(var_99,     color="#e74c3c", linestyle="--", linewidth=1.5,
                    label=f"VaR 99% = {var_99:+.4f}")
    ax_hist.axvline(cvar_99,    color="#c0392b", linestyle=":",  linewidth=1.5,
                    label=f"CVaR 99% = {cvar_99:+.4f}")
    ax_hist.set_xlabel("Daily Return")
    ax_hist.set_ylabel("Density")
    ax_hist.legend(fontsize=8)

    # Right: Metrics table
    ax_table.axis("off")
    rows  = list(metrics.keys())
    vals  = list(metrics.values())
    table = ax_table.table(
        cellText =[[v] for v in vals],
        rowLabels=rows,
        colLabels=["Value"],
        cellLoc  ="center",
        rowLoc   ="left",
        loc      ="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)

    # Style header row and row-label cells
    for (row, col), cell in table.get_celld().items():
        if row == 0 or col == -1:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f8f9fa")
        cell.set_edgecolor("#dee2e6")

    plt.tight_layout()
    plt.show()
