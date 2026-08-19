# Archive — superseded material

Nothing here is part of the current study. It is kept (not deleted) because this repo is not
under version control, so an archive is the only undo. Do not import from it, and do not cite
any number produced by it.

## Why each group was retired

**`notebooks/`** — the original exploratory notebooks and their code dumps. They carry the
retired "Physics-Informed / PINN" framing, which was never implemented: the actual method is an
L2 anchor toward a classical VaR prior, with no monotonicity or sub-additivity constraint.
They also predate the TRAIN/VAL/TEST protocol.

**`scripts/`** — `run_batch.py` and the notebook-patching helpers. The old batch driver executed
a notebook per ticker with a single seed and no validation block, so its hyperparameters were
effectively chosen on the test set. Replaced by `run_experiment.py` / `run_batch_anchored.py`.

**`outputs_stale/`** — every result produced before three corrections landed:
1. no VAL block (hyperparameters tuned on TEST),
2. single seed (no seed distribution),
3. the GARCH standardized-t quantile bug (inflated VaR by `nu/(nu-2)`; the BTC
   "−443% capital reserved" artefact).
Ranked on pass/fail and capital reserved rather than a strictly consistent loss.

**`images/`** — figures generated from those stale results.

**`scratch/`** — temp logs and throwaway plots.

## Replaced by

| Retired | Current |
|---|---|
| notebook batch pipeline | `run_experiment.py`, `run_batch_anchored.py` |
| pass/fail summary table | `evaluation/report.py` (pinball · DM · MCS · coverage gate) |
| ad-hoc backtest helpers | `evaluation/scoring.py` |
| implicit split, one seed | `evaluation/protocol.py`, `evaluation/harness.py` |
| hardcoded model branches | `models/registry.py` |
