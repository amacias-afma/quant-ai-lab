# Survey — disclosure practices in ML-for-VaR papers

Coding follows `paper/survey-protocol.md`, which was written **before any paper was read**.

**Coding discipline.** `not stated` means we read the relevant section and the practice was
absent. `unread` means we could not access or verify that section. Conflating the two would
manufacture a finding out of an access limitation — the same error this paper is about — so
they are kept separate and **only `not stated` counts as evidence**.

**Status: Editor condition E4 MET.** Five ML-VaR papers read in full and coded, plus one
ML-reproducibility paper (P5) read to test the §3.4 premise. A convenience sample of five cannot
characterise a literature and we do not present it as doing so.

**Headline: the survey falsified two of the draft's own premises and both were revised.** P4 does
held-out selection on a consistent loss and ensembles 20 networks; P5 proposes regularisation for
run-to-run variance and handles it correctly, with a degenerate-stability control we lacked. What
is absent in *every* paper read is narrower and more interesting: **test-reuse counting, power
analysis and pre-registration — 5 of 5 — and a strictly consistent scoring function for ranking,
4 of 5.**

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

**Now read in full.** The earlier partial read reached only the backtesting section; the full
article text was subsequently retrieved and §3 ("Data and Model Specifications") coded. The nine
`unread` fields below are now resolved. **The earlier version of this entry is superseded, not
deleted, so the change in coding is visible.**

| field | coding | evidence |
|---|---|---|
| `n_seeds` | **1 (fixed)** | *"We also set the random seed equal to 1."* Weights are re-initialised per forecast window, so the method is genuinely stochastic — one seed is a choice, not a property |
| `seed_dispersion_reported` | **no** | — |
| `selection_protocol` | **no held-out block** | epochs fixed at 300 by hand: *"it was difficult to choose an automatic threshold for the number of epochs to avoid overfitting, so each model was trained for 300 epochs."* Architecture ("a rather small architecture", LSTM-100 → 64 → 32 → 1) given without a selection procedure |
| `n_specs_disclosed` | **partial, and one arm withheld** | the `p ∈ {5,10,20,100}` grid is stated and all cells reported. But a separate reset-frequency arm was run and suppressed: *"Such approach has been proven faulty in results comparison, due to large jumps in volatility estimates. **The results are not reported here.**"* |
| `multiplicity_correction` | **not stated** | 3 indices x 4 sequence lengths x 3 distributions vs GARCH baselines, uncorrected |
| `test_reuse_stated` | **not stated** | — |
| `power_analysis` | **not stated** | 250 rolling forecasts per index — a small evaluation window, with no discussion of detectable effect size |
| `stability_claim` | **not found** | — |
| `consistent_scoring` | **no** | training uses negative log-likelihood; comparison uses the Lopez quadratic and Abad-Benito-Lopez loss functions (Abad et al. 2015). Neither is strictly consistent for a quantile |
| `coverage_gate` | **yes** | Kupiec (1995) and Christoffersen (1998) |
| `preregistered` | **no** | — |

**The withheld arm is the notable finding**, and it is disclosed in one sentence in passing. The
authors ran an experimental branch, judged it faulty, and reported neither its results nor how
many configurations it contained. Nothing here is concealed — the sentence is right there in the
text — but the specification count the paper implies is smaller than the specification count it
performed. That is precisely the integer §4 of our paper discloses, and the reason we disclose it.

## P6 — Data Driven VaR Forecasting using a SVR-GARCH-KDE Hybrid (Lux, Härdle & Lessmann, *Computational Statistics* 35:947-981, 2020; arXiv:2009.06910)

**Read in full.** Peer-reviewed journal article, not a preprint-only work.

| field | coding | evidence |
|---|---|---|
| `selection_protocol` | **HELD-OUT, CHRONOLOGICALLY PRIOR** | tuning period 2006-07-01 to 2011-06-30; forecasts evaluated 2011-07-01 to 2016-06-30. Tuned per index and per quantile separately |
| `consistent_scoring` | **NO — and this is the paper's weak point** | *"As criterion for selecting a model from grid search **and** evaluate the performance … the p-value of the test for conditional coverage is used … the one with the highest p-value is considered to be the best."* A coverage-test p-value is not a consistent scoring function, and the same criterion does selection and evaluation. The secondary loss is Lopez (1998), also not strictly consistent |
| `n_seeds` | **n/a** | SVR and KDE are deterministic given hyper-parameters. This paper is immune to our §3.4 by construction of the method, not by virtue of a control |
| `seed_dispersion_reported` | **n/a** | — |
| `n_specs_disclosed` | **derivable, not stated** | the grid is fully specified: `C` (9 values) x `psi` (10) x `gamma` (9) = **810 per index-quantile cell**, x 3 indices x 3 quantiles = **7,290**, each a moving-window refit. Better than most — the number can be computed — but it is never stated as an integer |
| `multiplicity_correction` | **partial, and honestly caveated** | Hansen's (2005) SPA test is used, which controls for data snooping within a comparison set. Running it with every model as benchmark reintroduces multiplicity, and they say so: *"Note that due to performing multiple statistical tests the p-values should be interpreted rather as an indication of model performance than in the context of a fixed significance level."* |
| `test_reuse_stated` | **not stated** | — |
| `power_analysis` | **not stated** | — |
| `stability_claim` | **not found** | — |
| `coverage_gate` | **yes** | full Christoffersen (1998) framework: unconditional, independence, conditional |
| `preregistered` | **no** | — |
| *(reproducibility artefact)* | **yes** | code published at quantlet.de |

**Two things worth separating.** The multiplicity caveat is better practice than most papers
manage — it downgrades their own p-values in advance rather than defending them. But selecting
on a **coverage-test p-value** is a real defect and a different one from any in our own study: a
model can achieve excellent conditional coverage while being far from the true quantile, because
the test only examines the sequence of violations. Ranking on it optimises calibration of the
hit sequence, not forecast quality.

## P3 — Quantile Convolutional Neural Networks for VaR Forecasting (Petneházi, arXiv:1908.07978)

**Read in full.** Likely the preprint of the *Machine Learning with Applications* paper listed
as a candidate earlier; **treated as one work, not two**, to avoid double-counting.

| field | coding | evidence |
|---|---|---|
| `n_seeds` | **not stated** | no mention of seeds, initialisations or repeated runs anywhere in the paper |
| `seed_dispersion_reported` | **no** | standard deviations *are* reported, but **across the 100 stocks**, not across seeds — a different quantity |
| `selection_protocol` | **not stated** | architecture is given (6 causal conv layers, 8 filters, kernel 2, 128 epochs, adadelta) with no selection procedure and **no validation block**: the split is 70% train / 30% test |
| `n_specs_disclosed` | **not stated** | — |
| `multiplicity_correction` | **not stated** | 100 stocks x 3 levels x 5 methods, reported as average rejection rates, uncorrected |
| `test_reuse_stated` | **not stated** | but a genuine robustness check is reported: "The experiments were repeated for the previous 10 years data (1999–2008) with a different set of randomly chosen stocks, and the results were quite similar" |
| `power_analysis` | **not stated** | — |
| `stability_claim` | **YES** | "the joint QCNN produced VaR exceedance rates with **consistently lower standard deviation** than the benchmark methods… which also justifies that this one produces the highest quality VaR estimates" |
| `stability_control` | **NONE** | no permutation, placebo or matched comparison |
| `consistent_scoring` | **no** | pinball is the *training* loss; results are reported as exceedance rates and DQ rejections. No pinball loss appears in any results table |
| `coverage_gate` | **yes** | Dynamic Quantile test (Engle & Manganelli) — a legitimate conditional-coverage test |
| `preregistered` | **no** | — |

**This is the pattern the survey was looking for: an explicit stability claim offered as
evidence of quality, with no control.**

Two things must be said fairly. First, the DQ evidence cited alongside it *is* legitimate —
fewer rejections is a real quality signal, and the stability sentence is not the paper's only
argument. Second, the paper is transparent about its design and reports a real robustness check
on a different decade and different stocks, which is more than many do.

> **Our conjecture, flagged as such.** The "joint" model is trained on all 100 stocks at once,
> so its per-stock forecasts are pulled toward a common fit. Lower *cross-stock* dispersion may
> therefore follow partly by construction — structurally the same mechanism as the shrinkage
> artefact in our §3.4, with pooling in place of an explicit penalty. **We have not tested this**
> and it is not a criticism we have established; it is a hypothesis that the same cheap control
> (refit with a scale-matched uninformative pooling target) would settle. We state it as a
> question, not a finding.

## P4 — Forecasting stock return distributions around the globe with quantile neural networks (arXiv:2408.07497)

**Read in full** (main text; Appendix D was not in the retrieved document).

**This paper contradicts our premise on most fields, and we report it as such.**

| field | coding | evidence |
|---|---|---|
| `selection_protocol` | **HELD-OUT VALIDATION BLOCK** | "We use the initial period from 1973 to 1994 for hyperparameter optimization. Models with varying hyperparameters are **trained from 1973 to 1989 and validated from 1990 to 1994**. The hyperparameters yielding the best validation performance… are then applied to train models from 1995 to 2018, generating out-of-sample predictions" |
| `consistent_scoring` | **yes** | selection and evaluation both on **average quantile loss** — a strictly consistent loss, not a proxy |
| `n_seeds` | **20, ensembled** | "As an additional form of regularisation, we use an **ensemble of 20 networks**" |
| `seed_dispersion_reported` | **no** | the ensemble *averages away* initialisation variation rather than reporting its spread — seeds are handled, dispersion is not shown |
| `multiplicity_correction` | **not stated** | Diebold–Mariano statistics are reported throughout; no correction across the many comparisons found in the main text |
| `n_specs_disclosed` | **unread** | "Detailed hyperparameter searches… are documented in Appendix D"; the appendix was not in the retrieved text, so we cannot verify whether a count appears |
| `test_reuse_stated` | **not stated** | — |
| `power_analysis` | **not stated** | — |
| `stability_claim` | **not found** | — |
| `preregistered` | **no** | — |
| `coverage_gate` | **unread** | evaluation focuses on quantile loss and DM; coverage tests not located in the retrieved text |

Also of note: the model is **refit annually on an expanding window with warm start from the
previous period's weights** — the same design our study used, arrived at independently.

**This is a counterexample to our §1 framing** and it is a strong one: held-out selection,
consistent scoring, ensembling, documented search, DM tests.

## P5 — On the Reproducibility of Neural Network Predictions (Bhojanapalli et al., Google Research, arXiv:2102.03349)

**Read in full** (main text; appendices not retrieved). Not a VaR paper. Read to answer Editor
condition H.6: *does anyone actually propose regularisation as a remedy for run-to-run variance,
and if so, do they offer the resulting stability as evidence of quality?*

**Answer: yes to the first, and emphatically no to the second. This paper does it correctly.**

They study *churn* — the fraction of test examples where two independently trained models
disagree — and propose two regularisers to reduce it: minimum-entropy regularisers, and a
co-distillation variant using a symmetric KL penalty between models' predicted distributions.
That is a shrinkage penalty aimed squarely at run-to-run variance, which is the practice §3.4
assumed existed. It exists.

What they do **not** do is treat the resulting stability as evidence that the method is better:

| control | present? | evidence |
|---|---|---|
| accuracy reported alongside stability | **yes** | every table pairs accuracy with churn, mean ± sd over 10 runs |
| the two axes explicitly separated | **yes** | *"one cannot infer churn from test accuracy, and understanding churn of an algorithm requires independent exploration"* |
| degenerate-stability control | **yes** | Table 1 includes a fully deterministic run: **churn 0.00 ± 0.00** at accuracy 91.76 ± 0.0, statistically indistinguishable from the 91.66 ± 0.12 baseline |
| tautology avoided in the metric | **yes** | *"for co-distillation we measure churn of a single model … **across independent training runs**"* — not between the two coupled models, which would be circular |
| dose–response with the cost axis shown | **yes** | Figure 3 plots the accuracy/churn trade-off as the penalty coefficient β varies |
| cost disclosed | **yes** | Table 4: the entropy regulariser *"predictably, increase[s] the calibration error"* |

They even state our §3.5 point in prose: *"multiple runs of a deterministic learning algorithm
produce models with zero churn, **independent of their accuracy**."* Stability is trivially
manufacturable, and they say so on page 3.

> **This falsifies the framing of §3.4 for a second time, and the protocol requires reporting
> it.** The claim that this practice lacks controls is false for the literature that studies
> stability *as its own objective*. That literature is careful.

**What survives, and it is narrower and better.** Our error was never *reporting* stability. It
was **inferring from stability a conclusion about the target** — that the anchor carried risk
information. Bhojanapalli et al. never make that inference: for them reduced churn *is* the
goal, a deployment desideratum in its own right, and accuracy is checked separately to confirm
nothing was lost.

The failure mode is what happens when the concept crosses domains without its controls. P3
(Petneházi) is the documented instance: *"consistently lower standard deviation than the
benchmark methods… **which also justifies that this one produces the highest quality VaR
estimates**."* There the inference is made explicitly, in a VaR paper, with no matched control.
We made the same one.

**So the honest structure of the argument is:** the reproducibility literature handles stability
correctly; a VaR paper borrowing the idea did not; and neither did we. **The controls did not
travel with the concept.**

---

## Tally

**Five ML-VaR papers, all read in full** (P1, P2, P3, P4, P6), plus one ML-reproducibility paper
read to test the §3.4 premise (P5, not counted in the VaR tally).

| field | `not stated / no` | `stated / done` | `n/a` |
|---|---|---|---|
| held-out selection block | 3 (P1, P2, P3) | **2 (P4, P6)** | — |
| **consistent scoring for selection** | **4 (P1, P2, P3, P6)** | **1 (P4)** | — |
| seeds handled (>1 run) | 3 (P1, P2, P3) | **1 (P4, ensemble of 20)** | 1 (P6, deterministic) |
| **seed dispersion *reported*** | **4** | **0** | 1 (P6) |
| number of specifications stated | 3 | **1 (P6, derivable)**, 1 partial (P2) | — |
| multiplicity correction | 4 | **1 (P6, SPA + explicit caveat)** | — |
| **test-set reuse count** | **5** | **0** | — |
| **power analysis** | **5** | **0** | — |
| **coverage gate** | **0** | **5** | — |
| **pre-registration** | **5** | **0** | — |
| stability claim made | 4 | 1 (P3) | — |
| matched control for that claim | 1 (none, P3) | 0 | — |

**Practice is heterogeneous and the two best papers are good in different ways.** P4 selects on a
strictly consistent loss and ensembles 20 networks; P6 has a chronologically prior tuning block,
a fully specified grid, Hansen's SPA test, published code — and then selects on a
conditional-coverage p-value, which P4 would not do. Neither dominates.

### What holds across all five

- **test-set reuse is never counted** — 5/5
- **no power analysis appears** — 5/5
- **no paper is pre-registered** — 5/5
- **seed dispersion is never reported** — 4/4 where applicable (P4 ensembles it away; P6's method
  is deterministic, so the question does not arise)
- **the coverage gate is universal** — 5/5. The one control this field has institutionalised.

### The finding we did not expect

**Only one of five ranks models on a strictly consistent scoring function for the quantile.**
The others use: exceedance rates and DQ rejections (P3), Lopez-family loss functions (P2),
and a conditional-coverage test p-value used simultaneously for selection and evaluation (P6).
A coverage test examines the *sequence of violations*; a model can pass it while sitting far
from the true quantile. This is a distinct failure from any in our own study — we ranked on
pinball throughout — and it is the most common defect in the sample.

## What we can and cannot say

**Can say (n = 5, all read in full):** practice is heterogeneous, and no single paper is careless.
Two use a held-out selection block; one selects on a consistent loss; one corrects for
multiplicity and caveats its own p-values; all five gate on coverage. But **none counts test-set
reuse, none reports power, none is pre-registered, and four of five rank on something other than
a consistent scoring function.** In one paper a stability claim is offered as evidence of quality
with no matched control.

**Cannot say:** anything about the field's base rate. Five papers chosen by search convenience is
an illustration, not a survey. We did not sample a frame, we did not pre-specify the search, and
we stopped when the Editor's condition was met — which is itself a stopping rule chosen after
seeing results. **By the standard §3.2 applies to this study's own nulls, this sample cannot
establish prevalence, and we do not claim it does.**

**Consequence for the draft — ACTED ON.** §1 previously read: *"Papers in this area typically
compare against a single benchmark, report a single seed, select hyper-parameters without a
held-out block."* **P4 falsifies the generality of that sentence.** §1 has been rewritten: it now
states that practice is heterogeneous, names P4 as the counterexample, and makes the narrower
claim that the controls which defeated this study (seed dispersion, test-reuse counting, power,
pre-registration) are absent *even in the paper that does the rest well*, while the coverage gate
is universal. §7 was updated in the same pass to source its "not standard practice" claim to this
survey rather than assert it.

This is recorded here because the finding ran against the draft's interest and the protocol
committed, before any paper was read, to reporting the tally whatever it showed.

## Editor condition E4 — status

**MET.** The Editor asked for three to five ML-VaR papers read in full; five are read in full and
coded, plus one ML-reproducibility paper (P5) read to test the §3.4 premise directly.

**The survey cost the draft two framings, both revised in its favour:**

1. **P4 falsified §1's original premise** that hyper-parameter selection is typically done
   without a held-out block. §1 was rewritten to say practice is heterogeneous.
2. **P5 falsified §3.4's premise** that regularisation-for-stability is proposed without
   controls. §3.4 and §3.5 were narrowed: the error is *inferring from stability a conclusion
   about the target*, not reporting stability.

Both revisions were committed to in advance by point 3 of the protocol, and both made the paper's
claims smaller. That is the correct direction of travel and it is worth stating plainly: a survey
that had confirmed everything we assumed would have been the less credible outcome.

### Residual limitations, stated rather than resolved

- **The sample is a convenience sample.** No frame, no pre-specified search string, and the
  stopping rule (the Editor's condition) was applied after seeing results.
- **P5 is not a VaR paper** and is excluded from the VaR tally, though it carries the most weight
  in the argument.
- **One lead not followed:** Madani et al. (2004), cited by P5 as using prediction disagreement
  *"as an estimate for generalization error and model selection."* If accurate that is an early
  instance of stability used as a quality signal. **Not read; recorded as open.**

## Sources

- P1 — LSTM-MDN, arXiv:2501.01278 — https://arxiv.org/pdf/2501.01278
- P2 — GARCHNet, *Computational Economics* (2023) —
  https://link.springer.com/article/10.1007/s10614-023-10390-7
- P3 — Quantile CNN, arXiv:1908.07978 — https://arxiv.org/pdf/1908.07978
- P4 — Quantile NNs around the globe, arXiv:2408.07497 — https://arxiv.org/pdf/2408.07497
- P5 — Bhojanapalli et al., *On the Reproducibility of Neural Network Predictions*,
  arXiv:2102.03349 — https://arxiv.org/pdf/2102.03349
- P6 — Lux, Härdle & Lessmann, *Computational Statistics* 35:947-981 (2020), arXiv:2009.06910 —
  https://arxiv.org/pdf/2009.06910
