# Project 01 — Anchored NN for Value-at-Risk

> **Does anchoring a quantile-loss neural network to a classical VaR prior improve its
> out-of-sample tail forecasts?**

## Research question

A neural network trained on the pinball (quantile) loss can, in principle, learn a sharper
one-day-ahead VaR than a fixed parametric or historical rule. In practice, with little tail
data and a single noisy loss signal, it can also wander into poorly-calibrated or degenerate
forecasts (e.g. a near-constant VaR that games unconditional coverage).

The **anchor** is a regularisation term that pulls the network's VaR toward a classical prior
(parametric-Normal or Historical VaR). The central question is an ablation:

> **Does the anchored NN beat the *same* network trained without the anchor — and does the
> anchor buy anything over the classical prior it is anchored to?**

If the anchored NN does not beat the unanchored NN, the anchor is doing nothing. If it does
not beat the classical prior, the network is not adding value over the rule it leans on. The
result is informative whichever way it lands.

> **Note.** An earlier version of this README described a "Physics-Informed NN" enforcing
> monotonicity in α and sub-additivity. No such constraints are implemented. The actual method
> is anchored quantile regression, and the model is named accordingly throughout.

## Method

The forecast target is the one-day-ahead return quantile at α ∈ {0.05, 0.01} (95% / 99% VaR).

- **Loss:** pinball loss `max((α−1)e, αe)` with `e = y − VaR`, plus an anchor penalty
  `weight · (VaR − VaR_prior)²`. `weight = 0` recovers plain quantile regression.
- **Architectures:** `SimpleQuantileNeuron` (a single linear unit — i.e. linear quantile
  regression, which is the *linear ablation*), `QuantileMLP` (the nonlinear rung) and
  `QuantileLSTM`. Registered in `models/registry.py`.
  **Note:** the linear neuron alone cannot support any claim about "neural networks" — the
  MLP-vs-neuron comparison on identical features is what licenses that language. Expect the
  linear spec's inter-seed IQR to be ≈ 0: pinball loss is convex, so every seed converges to
  the same optimum. Seed dispersion only becomes informative for the nonlinear rungs.
- **Anchors:** `Anchor NN` (parametric-Normal prior, `μ − z_α σ`) and `Anchor Hist NN`
  (rolling Historical VaR prior).
- **Why these priors are sensible:** the pinball-loss surface over (mean, std) multipliers is
  minimised near the classical `μ − z_α σ` point (see `functional_loss.png`), so anchoring to
  the parametric rule pulls the network toward the region the loss already prefers.

## Benchmark ladder

Reported in full — the ablation row is the most informative one.

1. **Trivial floor** — parametric-Normal VaR and rolling Historical VaR (fixed windows).
2. **Industry standard** — GARCH(1,1) with Student-t innovations (a benchmark, not the target
   of the paper).
3. **The ablation** — the unanchored NN: identical architecture and features, `weight = 0`.
4. **The models under test** — `Anchor NN` and `Anchor Hist NN`.

## Scoring (see `src/value_at_risk/evaluation/`)

- **Ranking:** mean **pinball loss** — the strictly consistent loss for a quantile. Models are
  ranked by loss, never by breach-rate pass/fail or "capital reserved".
- **Coverage gate:** **Kupiec** (unconditional) and **Christoffersen** (independence +
  conditional coverage). A model must pass both — a correct breach *count* with *clustered*
  breaches still fails.
- **Comparison:** **Diebold–Mariano** on loss differentials (HAC, bw = 5) for pairs; Model
  Confidence Set when comparing the whole ladder.

`scoring.py` and `protocol.py` are pure numpy/scipy and covered by golden tests:

```bash
cd 01_value_at_risk
pip install -e .            # required: src/ layout, makes `value_at_risk` importable
pip install -e ".[run]"     # + torch / arch / yfinance, needed to actually run the study
python -m pytest -q         # 50 passed (paths come from pyproject, no PYTHONPATH needed)
```

## Evaluation protocol (`protocol.py`)

- **Chronological TRAIN / VAL / TEST.** The anchor weight, rolling windows and architecture are
  chosen on VAL only; TEST is scored once. No shuffling.
- **Seeds.** NN results are a distribution over ≥ 10 seeds, reported as **median + IQR**, never
  the best seed. If the inter-seed IQR swamps the model-vs-benchmark gap, the honest conclusion
  is "no detectable difference".

## Status — honest gaps (work in progress)

The batch results in `outputs/` are an early single-seed pass and are **not yet the reported
numbers.** Before this is a defensible article:

- [ ] Rewire the notebook pipeline to the chronological VAL/TEST split in `protocol.py`
      (hyperparameters currently chosen without a held-out VAL block).
- [ ] Run the NN models over ≥ 10 seeds and report median + IQR.
- [x] Add pinball loss + Diebold–Mariano / MCS to the batch summary — tooling in
      `evaluation/{mcs,report,benchmarks}.py`, driven by `run_batch_anchored.py`
      (writes `outputs/anchored_batch_summary.{csv,md}`). **Outputs still need regenerating**;
      the old pass/fail `batch_summary.md` is deprecated.
- [x] Freeze and hash the input data — `data/snapshot.py` writes hash-verified CSV snapshots
      + manifest; `prepare()` reads them. **Run `python -m value_at_risk.data.snapshot` and
      commit `data/snapshots/` to pin the study's inputs.**
- [x] Fix the GARCH rescaling blow-up (e.g. BTC max reserve ≈ −443%) — the standardized-t
      quantile was divided by `sqrt((nu-2)/nu)` instead of multiplied; fixed in
      `garch_model.standardized_t_quantile`. **Outputs still need regenerating.**
- [ ] Disclose the two integers: specifications evaluated, and test-set evaluations.

## Layout

```
src/value_at_risk/
  models/
    registry.py              ← the only place that knows what architectures exist
    deep_var/
      architectures.py       SimpleQuantileNeuron (linear ablation), QuantileMLP, QuantileLSTM
      losses.py              AnchoredQuantileLoss  (pinball + w·(pred − prior)²)
      train.py               walk-forward training loop, registry-driven
      parametric_model.py    parametric & historical priors (the anchors)
      features.py            feature construction
    garch_model.py           GARCH(1,1)-t benchmark (fixed standardized-t quantile)
  evaluation/
    scoring.py               pinball · Kupiec · Christoffersen · Diebold–Mariano
    protocol.py              chronological split · multi-seed median+IQR
    harness.py               VAL-only weight selection · seed loop · CSV
    benchmarks.py            classical rungs as TEST-aligned forecasts
    mcs.py                   Hansen Model Confidence Set
    report.py                ranked ladder table (pinball · DM · MCS · gate)
run_experiment.py            one ticker × one α  → results CSV + meta.json
run_batch_anchored.py        panel × α levels    → anchored_batch_summary.{csv,md}
tests/                       43 tests, no torch required for the scoring/registry layer
_archive/                    superseded pipeline + stale results (see _archive/README.md)
```

### Data is frozen, not fetched

The first run downloads each ticker once and writes `data/snapshots/<TICKER>@<END>.csv` plus a
`manifest.json` recording its sha256, row count and date range. Every later run loads that file
and **verifies the hash**, so a rerun cannot silently train on different data. Commit the
snapshots and the manifest.

```bash
python -m value_at_risk.data.snapshot                    # freeze the panel
python -m value_at_risk.data.snapshot --verify           # check nothing drifted
python -m value_at_risk.data.snapshot --force            # deliberate re-freeze (visible in git)
```

Feature construction is frozen too — see `docs/features.md`, pinned by
`tests/test_features_contract.py`.

### Adding a model to test

1. Define the `nn.Module` in `deep_var/architectures.py`.
2. Add one `ModelInfo` entry in `models/registry.py` (declare `expects_sequence` and its
   ladder `rung`).
3. Run it — no other file changes:

```bash
python run_experiment.py --models QuantileMLP --ticker ^GSPC --alpha 0.05 --seeds 10
# nonlinearity ablation: same features, linear vs MLP
python run_batch_anchored.py --models SimpleQuantileNeuron,QuantileMLP --alphas 0.05,0.01
```

## References

- Engle (1982) — ARCH. Bollerslev (1986) — GARCH(1,1).
- Koenker & Bassett (1978) — quantile regression / pinball loss.
- Taylor (2019) - forecasting VaR/ES by a semiparametric asymmetric-Laplace (ES-CAViaR)
  approach. NOTE: an earlier version of this file described this paper as a quantile-loss
  neural network. That was wrong; corrected 2026-08-19. See paper/references.md.
- Kupiec (1995); Christoffersen (1998) — VaR backtesting.
- Diebold & Mariano (1995) — predictive-accuracy comparison.
