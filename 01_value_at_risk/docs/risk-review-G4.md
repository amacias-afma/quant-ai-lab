# Risk review — G4 gate, Project 01 (Anchored quantile VaR)

> **SUPERSEDED — see `docs/falsification-result-N2.md` and `docs/risk-signoff-G4-final.md`.**
> The variance-reduction claim discussed below was later shown to be **tautological**: an
> uninformative prior reduces inter-seed dispersion just as much (Wilcoxon p = 0.844). This
> document is retained unedited as a dated record of what was believed at the time; it is a
> *record*, not a summary, and correcting it would destroy the history the paper reports.



Reviewer role: **Quant Risk**. Mandate: break the result before the market does. This gate
carries a **veto** that the Lead cannot overrule, only remediate.

Reviewed: stage 1, stage 2, stage 2b, stage A pilot, stage B (MLP). Date: 2026-08-19.

## VERDICT: **VETO** on any out-of-sample claim.

The work is well engineered and the code is well tested. That is not what is being vetoed.
What is vetoed is the *epistemic status* currently attached to the numbers: they are being
described as out-of-sample results, and they are not.

---

## F1 — VETO. The test set no longer exists.

`standards/validation-protocol.md` §1: *"TEST is evaluated once, at the end, after the design
is frozen... If you go back to TEST a second time, the honest description is 'we now have a
validation set of two blocks and no test set.'"*

Measured from the run manifests:

| run | combos | test-set evaluations | tickers |
|---|---|---|---|
| stage 1 | 16 | 528 | 8 |
| stage 2 | 16 | 528 | 8 |
| stage 2b | 16 | 528 | 8 |
| stage B | 5 | 315 | 5 |
| **total** | | **1 899** | |

The same TEST block was scored **four times** on `^GSPC, BTC-USD, NVDA, SQM, CL=F` and three
times on the other three. Design decisions between runs (weight grid, selection rule,
architecture, ticker subset) were informed by the previous run's TEST outcomes.

**There is no untouched holdout left in this dataset. All 8 tickers have been scored on TEST.**

Consequence: every reported DM p-value, MCS membership and coverage statistic is conditioned
on prior inspection of the same block. They are validation statistics. Reporting them as
out-of-sample is the single most likely reason a referee rejects this paper.

## F2 — VETO. "Capacity hurts accuracy" does not survive seed noise.

The stage-B claim that the 8 641-parameter MLP is worse than the 4-parameter linear model
(4 of 5 tickers) was tested against the models' own inter-seed dispersion:

| ticker | gap (MLP − linear) | IQR linear | IQR MLP | gap > sum of IQRs |
|---|---|---|---|---|
| BTC-USD | −5.1e-05 | 2e-06 | 8.9e-05 | no |
| CL=F | +8.5e-05 | 1.0e-05 | 1.1e-04 | no |
| NVDA | +4.1e-05 | 1.0e-05 | 4.7e-05 | no |
| SQM | +1.6e-05 | 7e-06 | 4.0e-05 | no |
| ^GSPC | +9.1e-05 | 1.9e-05 | 1.0e-04 | no |

**0 of 5.** Every gap is smaller than the combined seed dispersion of the two models being
compared. The claim is unsupported and must be withdrawn.

## F3 — MAJOR. The MLP is a strawman, so any "capacity" conclusion is a defect.

The charter requires steelmanning. The MLP inherited `lr = 0.01` and `epochs = 500` from the
**linear** model. No learning rate, depth, width, batch scheme or regularisation was ever tuned
for it, on VAL or anywhere else. An 8 641-parameter network trained full-batch for at most 500
epochs at a learning rate chosen for a 4-parameter model is not a fair representative of
"higher capacity".

Aggravating: `epochs_per_block` is recorded in `train_model`'s history but **never persisted**,
so there is no evidence the MLP converged at all. Convergence is currently unfalsifiable.

Any statement of the form "capacity does not help / hurts" is a statement about *this
untuned configuration*, not about capacity.

## F4 — MAJOR. Nothing survives multiplicity.

Stage B alone: 20 comparisons. Holm and Bonferroni both leave **2 of 20** (uncorrected: 5).
Across the full study the count of comparisons is in the hundreds against 1 899 test-set
evaluations. No multiplicity correction has been applied anywhere in the reported results.

Worse, the surviving claims are **concentrated in one asset**: 3 of the 5 `claim_supported`
rows in stage B are SQM. A result driven by a single name in a 5-name panel is not a panel
result.

## F5 — MAJOR. Ticker selection was contaminated by TEST outcomes.

The stage-A/B panel core (`^GSPC, NVDA, BTC-USD`) was chosen to "span the three behaviours
observed in stage 1" — grid-edge, anchor-disabled, interior weight. Those behaviours are
**stage-1 TEST results**. There is a direct selection path from test outcomes to the design of
the follow-up study. Stage A being VAL-only does not repair this: the assets it ran on were
already chosen using TEST.

## F6 — MAJOR. The MCS has no discriminating power at this sample size.

In stage B all **9 of 9 models are inside the 90% Model Confidence Set in 5 of 5 combinations**
— including `Parametric-Normal`, which fails the coverage gate 5/5. When the MCS cannot
exclude a model that is demonstrably miscalibrated, it is not evidence of equivalence; it is
evidence that n is too small to conclude anything. Any ranking presented from these runs is
decorative.

## F7 — MINOR. Seed dispersion is reported without uncertainty.

Every variance claim rests on an IQR estimated from 10–20 seeds, reported as a point value with
no confidence interval and no bootstrap. The stage-A non-monotonicity that "disappeared" when
seeds went 10 → 20 is direct evidence that these estimates are unstable at the sample sizes
used. Ratios such as "16.1x" and "4.06x" should not be quoted without an interval.

---

## What survives

Risk is not obliged to leave nothing standing. Two findings are robust *because* they are
statements about the procedure rather than about out-of-sample performance:

1. **Anchor-weight selection is indistinguishable from chance.** 45% / 39% / 37% / ~40%
   across four independent configurations (coarse grid, fine grid, argmin, one-SE), none
   distinguishable from a coin flip, with expected value negative (average damage when wrong
   exceeds average gain when right). This replicates and does not depend on TEST being pristine
   — it is a claim about VAL→TEST transfer failure, which F1 only reinforces.
2. **The anchor reduces inter-seed dispersion, and the loss-based selection criterion discards
   it.** Consistent across every run. The mechanism is clean: pinball selection optimises the
   level, not the dispersion, so a stabiliser is switched off by construction. This is a
   methodological contribution and is not a performance claim.

## Required remediation before G5

1. **Relabel.** Every current result becomes "validation / exploratory". Remove the words
   out-of-sample, test-set and any DM p-value presented as confirmatory. Disclose all 1 899
   evaluations and the four passes over the block.
2. **Obtain a genuinely untouched holdout.** No asset in the current universe qualifies.
   Options: new tickers never loaded, or a later data window. Freeze it, do not look at it, and
   score the *final* frozen specification exactly once.
3. **Withdraw F2's claim.** "Capacity hurts" is out unless it can be shown to exceed seed noise.
4. **Either steelman the MLP or drop capacity claims entirely.** If kept: tune lr/epochs/width
   on VAL, persist `epochs_per_block`, and demonstrate convergence.
5. **Apply and report a multiplicity correction** on every family of comparisons.
6. **Bootstrap the IQR ratios** and report intervals.
7. **Disclose the ticker-selection path** from stage-1 TEST results to the stage-A/B panel.

## Note to the Lead

The honest paper here is *not* "we tested anchored neural VaR". It is **"a pre-registered
audit of hyper-parameter selection in VaR modelling, in which the selection signal proved
indistinguishable from noise across four configurations, and in which the one consistent
benefit of the regulariser is invisible to the selection criterion used to choose it."**

That paper does not need a pristine test set for its central claim, is supported by everything
above, and is more interesting than the result the study set out to find.
