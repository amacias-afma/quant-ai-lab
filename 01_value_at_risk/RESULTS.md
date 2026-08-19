# Results — Project 01

> **Every empirical claim this project produced has been withdrawn.** This document reports
> that outcome and what caused each withdrawal. An earlier version of this file headlined
> "the anchor sharply reduces seed variance" as a finding; that claim is **retracted** — see §3.

Status: G4 closed (`docs/risk-signoff-G4-final.md`). The deliverable is a methods /
negative-results paper, not a Value-at-Risk result.

---

## 1. What was asked

Does anchoring a quantile-loss model to a classical VaR prior — an L2 pull toward
`mu - z*sigma` or a rolling historical quantile — improve one-day-ahead VaR forecasts?

Panel: 8 tickers x alpha in {0.05, 0.01}, 10-year frozen snapshots, chronological
TRAIN/VAL/TEST, monthly refit, >= 10 seeds per specification.

## 2. What was found

**Nothing that survived examination.** Four claims were produced and all four were withdrawn:

| claim | why it fell | cost of the check |
|---|---|---|
| Anchoring lowers out-of-sample loss | 2/32 uncorrected, **1/26 after Holm**; and the test block had been scored four times, so no out-of-sample status remains | reading the run manifests |
| Selection error is indistinguishable from chance | **9.5% power**; MDE band [0.15, 0.85]; needs ~125 comparisons | a power calculation |
| The anchor reduces seed dispersion | **tautological** — an uninformative prior does it equally well | 28 min of validation compute |
| Higher capacity hurts accuracy | 0/5 gaps exceed seed noise; MLP never tuned; zero power at n = 5 | comparing gaps to IQRs |

## 3. The retraction that matters

The variance-reduction claim was the strongest thing this project produced. It replicated
across four independent runs (19/20, 21/23, 25/27, 15/16 comparisons), with median ratios of
13.5x–16.1x, a sign test at **p = 5e-04**, and a dose-response relationship between the anchor
weight and the effect size (**Spearman rho = +0.585, p < 1e-4, n = 85**).

It is an artefact of shrinkage. An L2 penalty pulls every seed toward the *same fixed target*,
so inter-seed dispersion must fall regardless of whether the target is sensible.

The decisive test used the real prior **permuted in time** — identical mean, standard deviation
and marginal distribution, correlation with tomorrow's tail ~0.0007 instead of 0.041:

| ticker | weight | real prior | shuffled prior |
|---|---|---|---|
| NVDA | 0.5 | 1.55x | 1.49x |
| NVDA | 1.0 | 2.25x | **2.91x** |
| SQM | 0.5 | 0.76x | 0.63x |
| SQM | 1.0 | 1.33x | 1.15x |
| ^GSPC | 0.5 | 3.05x | **3.10x** |
| ^GSPC | 1.0 | 12.47x | **16.76x** |

**Wilcoxon p = 0.844.** A prior containing no information reduces dispersion as much as the
real one. Details and the misleading-aggregate trap: `docs/falsification-result-N2.md`.

## 4. Disclosure

From the append-only ledger (`outputs/test_touch_ledger.jsonl`), not a hand count:

- **1 959 test-set evaluations**
- **16 ticker-alpha cells**, scored across **4 passes**, with design choices between passes
  informed by the previous pass's outcomes
- **0 cells scored only once**

Per `standards/validation-protocol.md` §1, the honest description is that this project has
several validation blocks and **no test set**. The 1 959 figure exceeds the 1 899 reconstructed
from run manifests: two early debugging runs appear in no manifest, which is precisely the gap
the ledger was built to close.

## 5. Three implementation defects, all favouring the hypothesis

Found and fixed during the study. Each, before correction, biased results toward the anchored
model:

1. **Early stopping.** The rule stopped when consecutive epochs changed by < 1e-6. Pinball loss
   on a linear model is piecewise linear and plateaus; adding the L2 anchor made the objective
   smooth. The anchored model therefore trained ~42x more epochs per block. *The anchor was
   buying training, not information.*
2. **Unequal training rows.** Rows with an undefined prior were dropped only when an anchor was
   active, giving `weight = 0` about 252 extra training rows. The weight grid was not comparing
   like with like.
3. **A Diebold-Mariano exception** on identical forecasts crashed 8 of 16 panel cells — exactly
   the cells where validation had switched the anchor *off*, i.e. the most informative outcome.

That all three errors ran the same direction is the paper's most transferable observation.

## 6. What is worth publishing

Not a VaR result. A documented case study of how a pre-registered quantitative study produces
findings that dissolve, and which control catches each:

| control | what it caught | cost |
|---|---|---|
| Golden tests on scoring conventions | sign-convention and coverage bugs | minutes |
| Multi-seed protocol | single-seed results were noise | small |
| Diebold-Mariano + Holm | 5 of 32 "wins" became 1 | free |
| Model Confidence Set | ladder is inseparable at this n | free |
| TEST-touch ledger | 1 959 evaluations, 4 passes, 60 uncounted | trivial |
| **Power analysis** | headline null had 9.5% power | free |
| **Nonsense-prior control** | strongest finding was arithmetic | 28 min |

The last two cost almost nothing and destroyed the two findings that everything else had
passed. That is the argument.

## 7. Reproducing

```bash
pip install -e ".[run]"
python -m pytest -q                              # 94 passed
python -m value_at_risk.data.snapshot --verify   # frozen inputs, sha256
python -m value_at_risk.evaluation.ledger --summary
```

| artefact | contents |
|---|---|
| `outputs/anchored_batch_*` | stage 1 panel (first-touch primary set) |
| `outputs/stage2*`, `stageB_mlp/` | exploratory re-runs — **not out-of-sample** |
| `outputs/nonsense_prior_test/` | the falsification test |
| `outputs/test_touch_ledger.jsonl` | every scoring pass |
| `docs/` | pre-registration amendments, Risk reviews, power analysis |
