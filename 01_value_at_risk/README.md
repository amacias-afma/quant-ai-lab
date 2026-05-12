# Project 01 — Deep VaR

> **Physics-Informed Neural Networks for Value-at-Risk forecasting**

## Research Question

Can a Physics-Informed Neural Network (PINN) outperform a properly-tuned GARCH(1,1) baseline at predicting the 95% and 99% Value-at-Risk of equity returns under realistic out-of-sample conditions?

## Approach

1. **Data:** [describe dataset — e.g., S&P 500 daily returns, 2010-2024].
2. **Classical baseline:** GARCH(1,1) with Student-t innovations, MLE-fitted via Nelder-Mead.
3. **Neural architecture:** Single-quantile PINN with a physics-informed regularization term enforcing risk-coherence (monotonicity in α, sub-additivity).
4. **Backtest:** rolling-window, walk-forward validation. Tested for breach frequency under Kupiec POF and Christoffersen conditional coverage.

## Key Findings

- The PINN matches but does not consistently dominate the GARCH baseline at the 95% / 99% quantiles in out-of-sample.
- More importantly: **predicting a single quantile leaves the portfolio blind to the shape of the tail past that quantile.** A 99% VaR breach can mean -1% or -50% — and the model cannot tell.
- This limitation motivated the migration to full **density forecasting** in [Project 02](../02_density_forecasting/).

## How to Run

```bash
cd 01_value_at_risk
jupyter lab
# open the notebook(s) in numerical order
```

## Files

- `value_at_risk.ipynb` — main file to run the process.

## References

- Engle (1982) — ARCH processes.
- Bollerslev (1986) — GARCH(1,1).
- Raissi et al. (2019) — Physics-Informed Neural Networks.
- Kupiec (1995); Christoffersen (1998) — VaR backtesting.
