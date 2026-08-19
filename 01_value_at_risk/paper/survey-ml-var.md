# Survey — disclosure practices in ML-for-VaR papers

Coding follows `paper/survey-protocol.md`, which was written **before any paper was read**.

**Coding discipline.** `not stated` means we read the relevant section and the practice was
absent. `unread` means we could not access or verify that section. Conflating the two would
manufacture a finding out of an access limitation — the same error this paper is about — so
they are kept separate and **only `not stated` counts as evidence**.

**Status: PARTIAL. This does not yet satisfy Editor condition E4.** Two papers, one read
substantially and one only in part. A convenience sample of two cannot characterise a
literature and we do not present it as doing so.

---

## P1 — LSTM Mixture Density Networks for risk forecasting (arXiv:2501.01278)

Read substantially (full preprint text retrieved and searched).

| field | coding | evidence |
|---|---|---|
| `n_seeds` | **1 (fixed)** | "seed-setting is fixed for the Python backend with the function `random.seed()`, for the numpy package via `numpy.random.seed()` and for TensorFlow backend via `tensorflow.rand.set_seed()`" |
| `seed_dispersion_reported` | **no** | a single fixed seed is used; no spread across initialisations is reported |
| `selection_protocol` | **manual, no held-out block stated** | "the procedure of determining a default architecture is done by **experimenting with different hyperparameters and manually observing the effects parameter changes have on model performance**" |
| `n_specs_disclosed` | **not stated** | the number of configurations tried during that manual exploration is not given |
| `multiplicity_correction` | **not stated** | no correction found |
| `test_reuse_stated` | **not stated** | no count of scoring passes |
| `power_analysis` | **not stated** | none found |
| `stability_claim` | not found in the sections read | — |
| `stability_control` | n/a | — |
| `consistent_scoring` | partial | evaluated by coverage tests; negative log-likelihood used for training |
| `coverage_gate` | **yes** | Kupiec (1995) POF, Christoffersen (1998) independence, and the joint conditional-coverage test are all implemented |
| `preregistered` | **no** | — |

**Note.** The authors are explicit and transparent about the manual procedure; they describe it
as a deliberate choice to present a *default* architecture rather than per-dataset optimisation.
Nothing here is concealed. **That is the point.** The practice is visible, reasonable on its own
terms, and still leaves the number of configurations undisclosed and the result resting on one
seed — the same position our study was in before its own controls were applied.

## P2 — GARCHNet: VaR forecasting with GARCH based on neural networks (*Computational Economics*, 2023)

**Partially read.** The retrieved text contains the backtesting section but appears not to
include the full training/implementation detail; methods-related terms returned only five
matches across the document.

| field | coding | evidence |
|---|---|---|
| `coverage_gate` | **yes** | Kupiec (1995) unconditional coverage and Christoffersen (1998) conditional coverage both described |
| `n_seeds` | **unread** | — |
| `seed_dispersion_reported` | **unread** | — |
| `selection_protocol` | **unread** | — |
| `n_specs_disclosed` | **unread** | — |
| `multiplicity_correction` | **unread** | — |
| `test_reuse_stated` | **unread** | — |
| `power_analysis` | **unread** | — |
| `stability_claim` | **unread** | — |
| `preregistered` | **unread** | — |

**P2 contributes one coded field and nine `unread`. It is not evidence for the premise.**

---

## Tally so far

| field | `not stated` | `stated` | `unread` |
|---|---|---|---|
| seed count > 1 | 1 | 0 | 1 |
| seed dispersion reported | 1 | 0 | 1 |
| held-out selection block | 1 | 0 | 1 |
| number of specifications | 1 | 0 | 1 |
| multiplicity correction | 1 | 0 | 1 |
| test-set reuse count | 1 | 0 | 1 |
| power analysis | 1 | 0 | 1 |
| coverage gate | 0 | **2** | 0 |
| pre-registration | 1 | 0 | 1 |

**The one field both papers report is the coverage gate** — the control the VaR literature has
long institutionalised. Everything our study found decisive is, in the single paper we could
read fully, absent.

## What we can and cannot say

**Can say (n = 1 read fully):** at least one recent ML-VaR paper selects architecture by manual
experimentation without a stated held-out block, does not disclose how many configurations were
tried, reports a single fixed seed, and applies no multiplicity correction — while correctly
implementing the classical coverage backtests.

**Cannot say:** anything about the field's base rate. Two papers, one partially read, is an
illustration, not a survey.

## To close E4 properly

1. **Five more papers read in full**, methods sections included, coded against the same schema.
   Preprints (arXiv) are preferable to paywalled versions purely for access.
2. **A stability-claim search specifically.** The premise about §3.4 needs papers that claim
   robustness or stability from a regulariser. One search result described an ML-VaR comparison
   in terms of "accuracy and stability" — that thread should be followed.
3. **Report the tally whatever it shows.** If these practices turn out to be common, the paper's
   §1 framing must change accordingly. That commitment is recorded here, before the remaining
   papers are read.

## Sources

- arXiv:2501.01278 — https://arxiv.org/pdf/2501.01278
- GARCHNet, *Computational Economics* — https://link.springer.com/article/10.1007/s10614-023-10390-7
  (open-access mirror: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10201522/)
