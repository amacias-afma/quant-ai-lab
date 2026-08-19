# Four Findings That Dissolved: A Pre-Registered Audit of a Value-at-Risk Study

**Draft v1** — G5. Every number is read from `outputs/paper_figures.json`, generated from the
result CSVs and the test-touch ledger. None is typed by hand.

---

## Abstract

We pre-registered a study of whether anchoring a quantile-loss model to a classical
Value-at-Risk prior improves one-day-ahead forecasts, and we report that it does not — but
that is not the contribution. Over eight assets and two quantile levels the study produced
four findings, each apparently well supported, and **all four were subsequently withdrawn**.
One fell to a multiplicity correction, one to a power analysis, one to a check of the model's
own seed noise, and one — the strongest, replicated across four runs with a dose–response
relationship and a sign test at p = 5.2e-04 — to a control that cost 28 minutes of validation
compute and no test data at all.

We disclose **1,959 test-set evaluations across 16 asset–level cells and four scoring passes**,
with **zero cells scored only once**, and we reproduce the decisive artefact synthetically with
ground truth known: across a ten-point weight grid an anchor carrying **no information**
stabilises the estimator **at least as much as the true optimum at 9 of 10 weights**
(sign test p = 0.021) while forecasting materially worse. **The two controls that destroyed the two
surviving findings — a power calculation and a scale-matched permutation — were also the two
cheapest, and neither is standard practice in this literature.** That is the paper's argument.

---

## 1. Introduction

The literature on machine-learning VaR reports improvements over econometric baselines with
some regularity. Papers in this area typically compare against a single benchmark, report a
single seed, select hyper-parameters without a held-out block, and rarely state how many
specifications were evaluated. Our aim was narrow: run one such comparison under a
pre-registered protocol and report whatever fell out.

What fell out was nothing — four times over, for four different reasons. The reasons are the
paper.

**Contributions.**

1. A pre-registered VaR study reporting a complete null, with dated amendments including
   several that weakened the authors' own position.
2. **Four failure modes observed in a single pre-registered study**, with the control that
   detects each and its cost. We claim instances, not coverage: one study cannot establish a
   taxonomy and we do not present one.
3. A demonstration — analytical and synthetic, with ground truth known — that
   **shrinkage-induced stability is not evidence of a good prior**, together with a
   scale-matched permutation control that separates the two for minutes of compute. The
   shrinkage mechanism is classical; that it is *reportable as an empirical finding*, and
   survives a pre-registered protocol, is what we document.
4. *(Software artefact, Appendix A.)* An append-only **test-touch ledger** that makes
   evaluation reuse countable rather than reconstructible after the fact.

## 2. Setup

**Question.** Does an L2 penalty pulling a quantile forecast toward a classical VaR prior
(rolling Normal `mu - z*sigma`, or rolling historical quantile) improve out-of-sample pinball
loss relative to the identical unanchored model?

**The estimator nests its own baseline.** The anchor weight grid includes zero, so the anchored
specification *contains* the unanchored one. It therefore cannot be worse by construction; it
can only lose through validation-selection error. This turns out to matter (§3.2).

**Data.** Eight instruments spanning equity index, single names, crypto, metals, energy and FX.
Ten-year frozen snapshots with sha256 manifests, verified on every load. Daily log returns.

**Protocol.** Chronological TRAIN / VAL / TEST; monthly refit on an expanding window with warm
start; ≥ 10 seeds per specification reported as median and IQR; pinball loss for ranking;
Kupiec and Christoffersen as a coverage gate; Diebold–Mariano with HAC errors for pairs;
Hansen's Model Confidence Set across the ladder.

**Primary evidence set.** Only two cells had ever been scored before the main panel
(`^GSPC` at both levels and `BTC-USD` at α = 0.05, during debugging). Excluding them leaves
**13 cells, 7 tickers, 26 comparisons** of genuine first contact. The exclusion criterion is
file timestamps, independent of any result; we verify below that it moves both headline numbers
*against* our thesis.

## 3. Four findings and their withdrawals

### 3.1 "Anchoring improves out-of-sample loss" — multiplicity

Three of 26 comparisons reject at 5% uncorrected. **One survives Holm.** More seriously, by the
time we reported this the test block had been scored four times, so no comparison retains
out-of-sample status.

*Control: a multiplicity correction and a count of test-set passes. Cost: free.*

### 3.2 "Weight selection is indistinguishable from chance" — power

Of 16 comparisons where validation selected a non-zero weight, **6 (37.5%) were worse than
`w = 0`** on the evaluation block. Binomial p = 0.45 against a coin flip; 95% CI
[0.15, 0.65]. We initially reported this as evidence that the selection signal is noise.

It is not. At n = 16 the design detects a proportion only below **0.147** or above **0.853**.
Power against the observed effect is **9.5%**. Reaching 80% power would need **125
comparisons**. A selection procedure with a genuinely useful 30% error rate would have been
missed 91% of the time.

The correct statement is *undetermined*, not *chance*. We also retract our earlier framing of
four configurations as independent replications: they share assets and periods.

*Control: a power calculation. Cost: free. Never performed until a reviewer demanded it.*

### 3.3 "Higher capacity hurts accuracy" — the model's own noise

An 8,641-parameter MLP lost to a 4-parameter linear model in 4 of 5 assets. **Zero of the five
gaps exceed the combined inter-seed IQR of the two models being compared.** The MLP also
inherited its learning rate and epoch budget from the linear model and was never tuned;
convergence was not recorded and so is unfalsifiable.

*Control: compare effect sizes to the models' own seed dispersion. Cost: free.*

### 3.4 "Anchoring stabilises the estimator" — tautology

This was the strongest result in the project. Across four runs the anchored estimator had lower
inter-seed IQR in 19/20, 21/23, 25/27 and **15/16** comparisons (primary set: p = 5.2e-04,
median ratio **13.5x, 95% CI 4.6–20.7**, n = 16), with a clean dose–response between the
selected weight and the effect:
**Spearman ρ = +0.585, p = 4.1e-09, n = 85**.

An L2 penalty pulls every seed toward the *same fixed target*. As the weight grows all seeds
converge on that target and inter-seed dispersion goes to zero **by construction** — whatever
the target is. The dose–response we took as corroboration is the signature of the artefact.

**The control.** Re-run with the real prior **permuted in time**: identical mean, standard
deviation and marginal distribution; correlation with tomorrow's absolute return 0.0007 versus
0.041. Matched on magnitude, stripped of information.

| ticker | weight | real prior | shuffled prior |
|---|---|---|---|
| NVDA | 0.5 | 1.55× | 1.49× |
| NVDA | 1.0 | 2.25× | **2.91×** |
| SQM | 0.5 | 0.76× | 0.63× |
| SQM | 1.0 | 1.33× | 1.15× |
| ^GSPC | 0.5 | 3.05× | **3.10×** |
| ^GSPC | 1.0 | 12.47× | **16.76×** |

The uninformative prior matches or beats the real one in 3 of 6 cells; **Wilcoxon p = 0.844**.
Bootstrapped over comparisons, the two are indistinguishable: real prior **1.9x (95% CI
1.0–7.8)** against shuffled **2.2x (95% CI 0.9–9.9)**. The intervals overlap almost entirely,
which is a more informative statement of the null than the rank test alone.

*Control: shrink toward a scale-matched nonsense target. Cost: 28 minutes, validation only,
zero test-set evaluations.*

**A trap.** Pooling all controls gives 1.32× against 2.36× for informative priors, which looks
like a genuine gap. It is produced entirely by a control shrinking toward an off-scale target,
which fights the data rather than shrinking within it. Only the **scale-matched** comparison is
diagnostic, and it is null. Reporting the aggregate would have preserved the claim.

### 3.5 The artefact isolated: a synthetic demonstration

One case does not establish that the artefact is general. We therefore reproduce it where the
ground truth is known and nothing is estimated from markets.

**Setup.** `y = X beta + eps`, `eps ~ N(0,1)`, so the optimal linear alpha-quantile is exactly
`X beta + z_alpha` and is available in closed form. We fit by descending
`pinball(y - X theta) + w * ||theta - a||^2` from random initialisations under a **finite step
budget** — the condition that produces seed dispersion in the first place, and the condition the
real study was under via early stopping. Three anchors: the **true** optimum; a **scale-matched
nonsense** vector with identical norm pointing elsewhere; and zero.

**Analytically**, the penalty's gradient is `2w(theta - a)`, so each step contracts the gap
between two runs that differ only in initialisation:

    spread_T  ~  spread_0 * (1 - 2*lr*w)^T

This depends on `w`, `lr` and `T`. **It contains no reference to `a`.** Shrinking toward the
truth and shrinking toward nonsense contract inter-seed dispersion by the same factor.

**Numerically**, over a ten-point log-spaced weight grid with 40 seeds per cell
(pure numpy, 22 seconds):

| weight | truth | nonsense | nonsense ÷ truth |
|---|---|---|---|
| 0.0005 | 1.05× | 1.09× | 1.03 |
| 0.0009 | 1.10× | 1.12× | 1.02 |
| 0.0016 | 1.13× | 1.17× | 1.04 |
| 0.0029 | 1.17× | 1.23× | 1.05 |
| 0.0053 | 1.26× | 1.28× | 1.02 |
| 0.0095 | 1.31× | 1.21× | **0.93** |
| 0.0171 | 1.51× | 1.67× | 1.11 |
| 0.0308 | 2.39× | 6.09× | 2.55 |
| 0.0555 | 5.46× | 17.81× | 3.26 |
| 0.1000 | 50.40× | 79.23× | 1.57 |

**Paired at each weight**, the uninformative anchor stabilises **at least as much as the true
optimum at 9 of 10 weights** (sign test **p = 0.021**), with a median relative ratio of
**1.04 (95% CI 1.0–1.8)**. The dose–response is emphatic and holds for every anchor
(Spearman ρ = **+0.974**, p = 1.3e-19, n = 30), reproducing the pattern we had taken as
corroboration in the real study (ρ = +0.585).

Loss moves the other way: the true anchor improves out-of-sample loss, the nonsense anchor
degrades it. **Stability tracks the penalty; usefulness tracks the target. They are different
axes, and only the second is evidence.**

> **A note on an earlier version of this section.** We first ran four weights and reported that
> the nonsense anchor "stabilises 2.5× more" than the truth. That figure was the largest of
> four cells. The denser grid shows the honest statement is *at least as much*, median 1.04 —
> weaker in magnitude, far stronger as evidence, because it is systematic rather than selected.
> We flag the change because quoting the most extreme cell is precisely the practice §3.1 and
> §3.4 criticise, and we did it in our own showcase before catching it.

**The paired column above is the paper's central point.** Stability rises with the penalty for
any target; usefulness depends on the target being right. A study that reports the first as
evidence for the second has reported an identity.

**Generalisation.** For any regularised estimator, stability under a shrinkage penalty is not
evidence that the shrinkage target is good. The mechanism is Stein/ridge shrinkage and is not
itself new; what we document is that it is **reportable as an empirical finding**, that it
survives a pre-registered protocol with a dose–response and p = 5.2e-04, and that a
scale-matched permuted control detects it for minutes of compute. That control should be
standard whenever stability is offered as a benefit.

**Every magnitude above carries a bootstrap interval.** The resampling unit is the
*comparison*, because the quantity quoted is a median across comparisons. A seed-level interval
would answer a different question and requires the per-seed losses, which our pipeline did not
originally persist — a gap we found only when trying to satisfy this requirement, and have since
closed. **The intervals are wide.** That is itself part of the finding: the magnitudes this
study reported were never as precise as a bare point estimate implies.

## 4. Disclosure

From the append-only ledger, not a hand count:

| quantity | value |
|---|---|
| test-set evaluations | **1,959** |
| asset–level cells | 16 |
| maximum scoring passes on one cell | **4** |
| cells scored exactly once | **0** |

Design choices between passes — weight grid, selection rule, architecture, asset subset — were
informed by the previous pass's outcomes. Per our own protocol the honest description is that
this project has several validation blocks and **no test set**.

The ledger figure exceeds the 1,899 we first reconstructed from run manifests: two early
debugging runs appear in no manifest. That 60-evaluation gap is precisely what an append-only
ledger exists to prevent, and it was found only because we built one.

## 5. Three defects, all in the same direction

1. **Early stopping.** The rule halted when consecutive epochs changed by < 1e-6. Pinball loss
   on a linear model is piecewise linear and plateaus; the L2 anchor makes the objective smooth.
   The anchored model trained roughly 42× more epochs per block. *The anchor bought training,
   not information.*
2. **Unequal training rows.** Rows with an undefined prior were dropped only when an anchor was
   active, giving `w = 0` about 252 extra rows. The weight grid was not comparing like with like.
3. **A Diebold–Mariano exception** on identical forecasts crashed 8 of 16 cells — exactly those
   where validation had switched the anchor *off*, the most informative outcome available.

Each defect, before correction, favoured the hypothesis under test.

**We must be careful here, because this is exactly the kind of claim this paper is about.**
Three of three in the same direction is **p = 0.125** one-sided under independent signs. By our
own standard in §3.2 — where 6/16 at p = 0.45 was ruled insufficient — n = 3 cannot support a
conclusion, and we do not draw one.

What we offer instead is a mechanism worth testing by someone with a larger sample: *a defect
that contradicts your hypothesis gets investigated; one that confirms it gets shipped.* Each of
these three was found only because an unrelated check failed, not because anyone audited a
result that looked right. That asymmetry in scrutiny is measurable in principle — count, across
a corpus of projects, the direction of defects found before versus after publication — and we
suggest it be measured rather than asserted.

## 6. What we verify about our own analysis

The primary evidence set is a post-hoc subset, which invites the obvious objection. Its
selection criterion is mechanical (file timestamps) and result-independent, and it moves both
headline numbers *against* our thesis:

| | full stage 1 | first-touch subset | direction |
|---|---|---|---|
| selection error | 45% | **38%** | further from 50% — weakens "selection is noise" |
| seed-IQR reduction (median) | 16.1× | **13.5×** | smaller — weakens "the anchor stabilises" |

We report this table so a reader can run the same check.

**Residual contamination.** The seven first-touch assets were never *scored* before, but the
code that scored them was revised after inspecting `^GSPC`'s output — that is how defects 1 and
2 were found. This is weaker than a pristine holdout and we say so rather than claiming
otherwise.

## 7. The taxonomy

| control | what it caught | cost |
|---|---|---|
| Golden tests on scoring conventions | sign-convention and coverage bugs | minutes |
| Multi-seed protocol | single-seed results were noise | small |
| Diebold–Mariano + Holm | 3 of 26 raw "wins" became 1 | free |
| Model Confidence Set | ladder inseparable at this n | free |
| Test-touch ledger | 1,959 evaluations, 4 passes, 60 uncounted | trivial |
| **Power analysis** | headline null had 9.5% power | **free** |
| **Nonsense-prior control** | strongest finding was arithmetic | **28 min** |

The two controls that destroyed the two surviving findings are also the two cheapest, and
neither is standard practice in this literature.

## 8. Limitations

- Eight assets, two levels, one market regime; conclusions about VaR modelling do not follow
  and are not offered.
- No untouched holdout exists in this universe. All results are validation-grade.
- The capacity arm rests on an untuned network and is withdrawn, not resolved.
- The nonsense-prior control was run on three assets at one level; it is decisive about
  mechanism, not about magnitude.
- We report a case study, not a survey. We do not claim these four failure modes are the most
  common ones, only that all four occurred in a single well-intentioned study.

## 9. Conclusion

We set out to test whether anchoring helps a quantile model forecast tail risk. It does not, in
our hands — but we cannot even claim that cleanly, because by the time we could ask the
question properly we had spent the test set four times over.

What we can offer is the record: four findings, four withdrawals, and the specific cheap check
that dissolved each. The most instructive is the last. A result replicated across four runs,
with a dose–response relationship and p = 5.2e-04, was a restatement of the fact that shrinkage
shrinks. Nothing in our protocol — pre-registration, multiple seeds, consistent scoring,
Diebold–Mariano, the Model Confidence Set, a coverage gate — was capable of catching it. Only
asking *what would this look like if the effect were mechanical?* was.

---

### Reproduction

```bash
pip install -e ".[run]"
python -m pytest -q                                  # 94 passed
python -m value_at_risk.data.snapshot --verify       # frozen inputs, sha256
python -m value_at_risk.evaluation.ledger --summary  # the disclosure integers
```

### Open items before submission

- [ ] Bootstrap intervals for every ratio quoted (Risk F7).
- [ ] Figure 1: dose–response, weight vs IQR ratio, ρ = +0.585.
- [ ] Figure 2: real vs shuffled prior, the six paired cells.
- [ ] Related work: pre-registration in finance; shrinkage estimators; hyper-parameter
      selection under small validation blocks.
- [ ] Decide venue framing: methods paper vs negative-results paper.
