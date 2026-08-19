# Project 02 — Density Forecasting & The Parametric Ceiling

> **Forecasting the entire return distribution with Neural SDEs and Path Signatures**

## Research Question

How do continuous-time Neural Stochastic Differential Equations (Neural SDEs) augmented with Path Signatures compare to highly-optimized parametric baselines (GARCH, Student-t, VIX-Scaled Student-t) at forecasting the **full conditional density** of next-day returns?

## Approach

### Classical Baselines (built and benchmarked first)
1. **GARCH(1,1) with Student-t innovations** — the workhorse.
2. **MLE-fitted Student-t** with constant ν.
3. **Proprietary factor-normalized VIX-Scaled Student-t** — a baseline that uses VIX as an exogenous scaling factor for the Student-t scale parameter, capturing volatility-of-volatility effects without leaving the parametric world.

### AI Architecture
4. **Path Signatures** (Rough Path Theory) as features encoding the full geometry of the recent price path, including non-linear cross-effects classical models cannot reach.
5. **Neural SDE** trained to map signatures → conditional density of next-day returns. Drift and diffusion are parameterized by neural networks; loss is the negative log-likelihood under the implied density.

### Evaluation
- Continuous Ranked Probability Score (CRPS) for full density quality.
- Quantile Score across the 1%, 5%, 95%, 99% quantiles.
- Kolmogorov-Smirnov and Anderson-Darling tests on PIT residuals (calibration).
- Tail-shape diagnostics under regime breaks.

## Status

✅ Classical baselines complete and benchmarked
🚧 Neural SDE implementation in progress
⏳ Out-of-sample evaluation pending

## How to Run

```bash
cd 02_density_forecasting
jupyter lab
```

## Files

- `01_classical_baselines.ipynb` — GARCH, Student-t, VIX-Scaled Student-t.
- `02_signatures_features.ipynb` — Path Signatures feature extraction.
- `03_neural_sde.ipynb` — Neural SDE architecture and training (WIP).
- `04_evaluation.ipynb` — calibration and tail diagnostics (WIP).

## References

- Lyons (1998) — Path Signatures.
- Chen et al. (2018) — Neural ODEs / SDEs.
- Kidger et al. (2021) — Neural SDEs as infinite-dimensional GANs.
- Gneiting & Raftery (2007) — CRPS scoring.