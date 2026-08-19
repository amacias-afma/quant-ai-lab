# Power analysis — Risk condition R2

> **SUPERSEDED — see `docs/falsification-result-N2.md` and `docs/risk-signoff-G4-final.md`.**
> The variance-reduction claim discussed below was later shown to be **tautological**: an
> uninformative prior reduces inter-seed dispersion just as much (Wilcoxon p = 0.844). This
> document is retained unedited as a dated record of what was believed at the time; it is a
> *record*, not a summary, and correcting it would destroy the history the paper reports.



Every null claim in this project, with the minimum effect the design could have detected.
Produced by `evaluation/power.py` (exact binomial; HAC-based normal approximation for DM).
Primary evidence set = stage 1 excluding `^GSPC` (both alphas) and `BTC-USD` alpha = 0.05.

**Convention.** A null is *informative* only if the observed value lies outside the design's
minimum detectable effect (MDE) band. Inside the band, "we did not reject" means the design
could not have rejected, whatever the truth.

---

## N1 — "Anchor-weight selection is indistinguishable from chance" — **UNINFORMATIVE**

| quantity | value |
|---|---|
| observed | 6/16 = **0.375** |
| p vs 0.50 | 0.454 |
| 95% CI | [0.152, 0.646] |
| **MDE at n = 16, 80% power** | **detectable only if <= 0.147 or >= 0.853** |
| power at the observed effect | **9.5%** |
| n needed for 80% power | **125 comparisons** |

The observed 37.5% sits deep inside the blind zone. At n = 16 this design could only have
detected a selection error below 15% or above 85% — i.e. only a procedure that was either
near-perfect or near-perfectly-wrong. **A selection error of 30%, which would be a genuinely
useful procedure, would have been missed 91% of the time.**

This is the study's headline null and it does not survive R2. It must be reported as:

> We could not distinguish the selection error from chance, but the design had 9.5% power
> against the observed effect and would require ~125 comparisons to reach 80%. We cannot
> exclude that anchor-weight selection carries real signal.

The four-configuration replication (45%, 39%, 37%, ~40%) does **not** repair this: those runs
share tickers and periods, so they are not independent replications. Pooling them would need a
model for that dependence, which we do not have.

## N2 — "The anchor reduces inter-seed dispersion" — **INFORMATIVE**

| quantity | value |
|---|---|
| observed | 15/16 = **0.94** |
| p vs 0.50 | **5.2e-04** |
| MDE band | <= 0.147 or >= 0.853 |
| informative | **yes** — 0.94 clears the upper MDE |

The only claim in the project whose effect is large enough for this design to see. It is
adequately powered *because the effect is near-total*, not because n is adequate. Reported as a
sign test; the magnitude (median ratio 13.5x) still needs the bootstrap interval required by
Risk F7.

## N3 — "Higher capacity worsens accuracy" — **UNINFORMATIVE, and already withdrawn**

| quantity | value |
|---|---|
| observed | 4/5 = 0.80 |
| p vs 0.50 | 0.375 |
| **MDE at n = 5** | **none — no effect size reaches 80% power** |

At n = 5 the design cannot detect *any* departure from chance at 80% power. Combined with Risk
F2 (0/5 gaps exceed combined seed IQR) and F3 (the MLP was never tuned), this claim is
withdrawn on three independent grounds.

## N4 — DM loss-differential nulls — **UNDERPOWERED at the observed effect sizes**

Observed |edge| in the primary set: median 2.6e-05, max 1.7e-04, against a typical pinball loss
of ~1e-03 — effects of roughly 1–8% of the loss level. With ~750 TEST observations per cell and
HAC standard errors, only the largest of these is detectable; 1 of 26 cleared Holm.

Reported as: *the study can detect anchoring effects of roughly the size of the largest observed
one, and is blind to the typical one.*

---

## Consequences for the paper

1. **N1 must be restated.** "Selection is noise" becomes "we could not detect selection signal,
   at 9.5% power". This weakens the paper's headline considerably and it is the honest version.
2. **N2 is the only claim carrying its own weight**, and it is a statement about estimator
   stability, not about forecast accuracy.
3. **N3 is withdrawn.**
4. **A confirmatory holdout sized at ~16 comparisons is pointless.** It would reproduce exactly
   the blind zone above. Either the holdout is sized for the effect we care about, or the
   confirmatory step is dropped and the paper stands as exploratory.

### Holdout sizing, if the confirmatory step is kept

To reach 80% power against a selection error of:

| true error | comparisons needed |
|---|---|
| 0.20 | ~40 |
| 0.30 | ~125 |
| 0.35 | ~300+ |

The plan's 6 tickers x 2 alphas x 2 anchors = 24 comparisons reaches none of these. Options:

- **(a) Widen** to ~20 tickers x 2 alphas x 2 anchors = 80 comparisons — powered against 0.25,
  still not against 0.30. Cost at the measured rate is prohibitive (days).
- **(b) Change the estimand** to something the design can see, e.g. the *paired* per-cell loss
  differential rather than a win/lose count, which uses magnitude instead of discarding it.
- **(c) Drop the confirmatory claim** and publish as an exploratory, pre-registered audit with
  the power analysis included as a contribution in its own right.

**Recommendation: (b) then (c).** The sign test throws away most of the information in each
comparison; a paired magnitude test on the same data would be materially better powered at no
extra compute. Failing that, (c) is honest and still publishable.
