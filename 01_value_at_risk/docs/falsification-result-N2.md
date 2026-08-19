# Falsification result — N2 (variance reduction) is tautological

Run: `run_nonsense_prior_test.py --tickers ^GSPC,NVDA,SQM --alpha 0.01 --seeds 10`
Validation only. **Zero test-set evaluations.** Data: `outputs/nonsense_prior_test/`.

Pre-registered decision rule (fixed in code before the run): if uninformative controls shrink
inter-seed dispersion comparably to the real priors, the variance reduction is a property of
shrinkage and every variance claim is withdrawn.

## Result

Median IQR reduction ratio (unanchored / anchored):

| prior | w = 0.5 | w = 1.0 | informative? |
|---|---|---|---|
| hist | 2.47 | 5.81 | yes |
| param | 1.55 | 2.25 | yes |
| **shuffled** | **1.49** | **2.91** | **no — same scale, no information** |
| constmean | 0.96 | 2.23 | no |
| zero | 0.79 | 0.68 | no |

### The decisive comparison: `param` vs `shuffled`

`shuffled` is the same vector as `param`, permuted in time. Identical mean, identical standard
deviation, identical marginal distribution; correlation with tomorrow's absolute return ~0.0007
versus 0.041 for the real prior. It is matched on magnitude and stripped of information.

| ticker | weight | param | shuffled |
|---|---|---|---|
| NVDA | 0.5 | 1.55 | 1.49 |
| NVDA | 1.0 | 2.25 | **2.91** |
| SQM | 0.5 | 0.76 | 0.63 |
| SQM | 1.0 | 1.33 | 1.15 |
| ^GSPC | 0.5 | 3.05 | **3.10** |
| ^GSPC | 1.0 | 12.47 | **16.76** |

Shuffled matches or exceeds the real prior in **3 of 6** cells. **Wilcoxon p = 0.844.**

**A prior with no information whatsoever reduces seed dispersion exactly as much as the real
one.** N2 is confirmed tautological: the reduction is a property of pulling every seed toward a
fixed point, not of the point being a sensible VaR.

## A trap in the aggregate figures

Pooling all controls gives 1.32x versus 2.36x for the informative priors, which looks like a
gap. **That comparison is misleading and must not be reported.** It is driven entirely by
`zero`, which shrinks toward an off-scale target: it fights the data rather than shrinking
within it, and it degrades the fit badly (^GSPC median VAL loss 4.64e-04 -> 6.28e-04;
NVDA 1.08e-03 -> 2.51e-03). `zero` is a broken control, not evidence of informativeness.

Only the **scale-matched** comparison is diagnostic, and it is null.

Supporting detail: `hist` shows the largest ratio (5.81) and is also the *least variable*
target (sd 0.00098 vs 0.00389 for `param`). A tighter shrinkage target produces more shrinkage.
Mechanical, again.

## Consequences

1. **Withdraw every variance claim.** "Anchoring stabilises the estimator" is out of the paper,
   in all four runs where it was reported (19/20, 21/23, 25/27, 15/16 and the 13.5x–16.1x
   medians). What remains sayable is "shrinkage shrinks", which is textbook.
2. **The capacity story (H5) loses its mechanism.** H5b — "the anchor absorbs more dispersion at
   higher capacity" — measured the same tautology at four model sizes. Withdrawn.
3. **No substantive empirical finding survives** across the whole project. Confirmed:
   - anchoring does not improve loss (1/26 after Holm, no test set);
   - selection error is *undetermined*, not chance (9.5% power);
   - variance reduction is tautological (this document);
   - capacity effects withdrawn on three independent grounds.

## What this episode is worth

The claim survived four independent runs, 85 comparisons, a Spearman correlation of +0.585
between weight and effect size, and a sign test at p = 5e-04. It was defeated by a control that
cost 28 minutes of validation compute and no test data at all.

That is the paper: **a statistically strong, replicated, mechanically inevitable result** — and
the single cheap control that dissolves it. The lesson generalises well beyond VaR to any
regularised model whose stability is reported as a benefit.

## Code note

The run crashed at the summary line: `informative` is object dtype (the baseline row carries
`None`), so `~series` performed a bitwise complement and produced integer indices. All fits and
the CSV completed first, so the result was recovered without re-running. Fixed by casting to
bool before negating.
