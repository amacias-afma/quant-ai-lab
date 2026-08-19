# Feature specification — 01-nn-var (Anchored NN for VaR)

**FROZEN.** Referenced by the pre-registration (`quant-articles/projects/01-nn-var/02-research/
hypotheses.md`). Changing anything here creates a new specification: it requires a dated
amendment in `hypotheses.md`, and it increments the *specifications evaluated* disclosure
integer. Do not add a feature after seeing results.

> **Canonical location.** This file belongs at `projects/01-nn-var/02-research/features.md` in
> the quant-articles repo, next to `hypotheses.md`. It is kept here so it lives beside the code
> that implements it; copy it across so the pre-registration reference resolves.

Frozen: 2026-08-16   Frozen by: Alvaro Macías   Implementation: `models/deep_var/features.py`

---

## 1. Input data

Daily close prices from a **frozen snapshot** (`data/snapshots/<TICKER>@<END>.csv`, sha256 in
`manifest.json`, verified on every load). Source: yfinance, 10-year window per ticker.
Returns are log returns:

    r_t = log(P_t) - log(P_{t-1})

The first row is dropped (undefined return). No other cleaning, no outlier removal, no
winsorising — a VaR study must not delete its own tail.

## 2. The feature set (identical for every model on the ladder)

Rolling window **w = 22** trading days (~1 month) for all rolling statistics.

| # | Feature | Definition | Rationale |
|---|---------|-----------|-----------|
| 1 | `log_ret` | r_t | the most recent shock; sign and size |
| 2 | `std_22d` | rolling std of r over w=22 | volatility state — the dominant driver of a conditional quantile |
| 3 | `mean_22d` | rolling mean of r over w=22 | local drift; the μ in μ − z·σ |

Three features, chosen to mirror the classical parametric prior `μ − z_α·σ` so that the
network is given exactly the inputs the anchor is built from. This is deliberate: it makes the
anchored-vs-unanchored comparison a clean test of the *anchor*, not of an information
advantage. It also makes `SimpleQuantileNeuron` a like-for-like linear ablation.

**Identical features across every rung.** The linear neuron, the MLP, the LSTM and the anchored
variants all receive the same three columns. Any performance difference is therefore
attributable to the model or the anchor, never to the inputs.

## 3. Target

    y_t = r_{t+1}

The model predicts the α-quantile of the **next** day's return.

## 4. No look-ahead — the alignment argument

Every feature at row *t* is computed from returns up to and including *t*; the target is
`r_{t+1}`, obtained by `shift(-1)`. So the forecast for day *t+1* uses only information
available at the close of *t*.

Rows containing any NaN are dropped **after** feature construction and **before** splitting.
This removes the first `w` rows (rolling warm-up) and the final row (no next-day target). The
anchor prior uses a 252-day historical window, so anchored specs additionally drop rows where
the prior is undefined; that filter is applied to X, y and dates positionally together, so
alignment is preserved.

Scaling: **none**. No standardisation is fitted, so there is no possibility of leaking test-set
moments into training. (If scaling is ever added it must be fitted on TRAIN only and applied
forward — that would be an amendment.)

## 5. Deliberately excluded

`features.py` also implements `log_ret^2`, `variance`, `historical_var`, `historical_var_2`,
`skewness`, `kurtosis`. **None are used.** They are excluded to keep the feature set minimal
and matched to the anchor's parametric form. Enabling any of them is a new specification and
must be pre-registered by amendment before results are seen — not selected afterwards because
it improved a number.

`historical_var` is additionally excluded from the *feature* set because Historical VaR already
enters the study as a benchmark rung and as the `hist` anchor prior; using it as an input as
well would blur what the anchored comparison is measuring.

## 6. Discretionary choices and where they are tested

| Choice | Frozen value | Robustness |
|--------|--------------|------------|
| rolling window w | 22 | sensitivity check reported in Robustness |
| feature count | 3 | fixed; expansion requires amendment |
| return type | log | fixed |
| scaling | none | fixed |

## 7. Amendments

(none)
