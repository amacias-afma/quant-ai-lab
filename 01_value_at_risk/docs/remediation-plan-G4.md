# Remediation plan — response to the G4 Risk veto

> **SUPERSEDED — see `docs/falsification-result-N2.md` and `docs/risk-signoff-G4-final.md`.**
> The variance-reduction claim discussed below was later shown to be **tautological**: an
> uninformative prior reduces inter-seed dispersion just as much (Wilcoxon p = 0.844). This
> document is retained unedited as a dated record of what was believed at the time; it is a
> *record*, not a summary, and correcting it would destroy the history the paper reports.



Author: **Quant Lead**. Status: proposed, awaiting Risk sign-off.
Risk's veto (`docs/risk-review-G4.md`) is accepted in full. Nothing below disputes a finding.

---

## 0. Position

The veto is correct and I am not contesting any of F1–F7. The study cannot claim out-of-sample
performance. What follows salvages what is actually supported, discards what is not, and buys
back a genuine holdout at the smallest defensible cost.

The paper is reframed exactly as Risk recommended:

> **A pre-registered audit of hyper-parameter selection in VaR modelling** — the selection
> signal proved indistinguishable from noise across four configurations, and the one consistent
> benefit of the regulariser is invisible to the criterion used to select it.

Performance of anchored quantile models is no longer the claim. It becomes context.

---

## 1. The one thing Risk did not weigh: TEST contamination is not uniform

Before the stage-1 panel, only two cells had ever been scored:
`^GSPC` (both alphas) and `BTC-USD` alpha = 0.05, from the debugging runs.
For the remaining cells, **stage 1 was genuine first contact**.

**Primary evidence set (strict first-touch):** stage 1, excluding `^GSPC` (both alphas) and
`BTC-USD` alpha = 0.05.

- 13 ticker-alpha combinations, 7 tickers, 26 anchored-vs-unanchored comparisons
- anchor disabled by VAL: 10/26
- selection error: 6/16 active = **38%**, binomial p = 0.454 vs chance, **95% CI [0.15, 0.65]**
- seed-IQR reduction: **15/16**, median **13.5x**
- Diebold-Mariano: 3/26 uncorrected, **1 survives Holm**

Both surviving findings hold on this set, and no performance claim does.

**Residual contamination, disclosed not hidden.** Even here the *specification* was shaped by
looking at `^GSPC`'s output: the early-stopping and row-alignment defects were found that way.
So these 7 tickers were never *scored* before, but the code they were scored with was revised
after a TEST look. This is weaker than a pristine holdout and the paper must say so in those
words. It is the reason section 4 exists.

## 2. Immediate actions — no new compute

| # | Action | Addresses |
|---|---|---|
| 2.1 | Relabel stage 2, 2b, A, B as **exploratory**. Remove "out-of-sample" and "test set" from all of them. | F1 |
| 2.2 | Designate the §1 first-touch subset as the **primary** evidence set; state its residual contamination. | F1 |
| 2.3 | Disclose **1 899 test-set evaluations** and four passes over the block, in the abstract, not a footnote. | F1 |
| 2.4 | **Withdraw** "capacity hurts accuracy". 0/5 gaps exceed combined seed IQR. | F2 |
| 2.5 | **Withdraw** all capacity claims, or relabel them "one untuned configuration". | F3 |
| 2.6 | Report Holm-corrected results everywhere; state that 1/26 survives on primary. | F4 |
| 2.7 | Disclose that 3 of 5 stage-B `claim_supported` are one asset (SQM). | F4 |
| 2.8 | Disclose the ticker-selection path from stage-1 TEST outcomes to the stage-A/B panel. | F5 |
| 2.9 | Stop presenting MCS membership as evidence of equivalence; report it as "no discriminating power at n = 5". | F6 |

## 3. Code work — small, testable

| # | Work | Addresses |
|---|---|---|
| 3.1 | Add Holm / Benjamini-Hochberg to `evaluation/scoring.py`; wire into `report.ladder_summary` and the verdicts table. Golden tests. | F4 |
| 3.2 | Bootstrap CIs for the seed-IQR ratio (percentile bootstrap over seeds); report interval, never a bare point. | F7 |
| 3.3 | Persist `epochs_per_block` into the run meta so convergence is auditable. | F3 |
| 3.4 | Record `lr`, `epochs`, `hidden_size`, `num_layers` per spec in the meta. Currently unauditable from outputs. | F3 |
| 3.5 | Add a **TEST-touch ledger**: an append-only JSON that logs every scoring pass (spec, ticker, alpha, timestamp, code hash). Make reuse impossible to lose track of again. | F1 |

3.5 is the structural fix. The reason we reached 1 899 evaluations is that nothing counted
them across runs.

## 4. Buying back a real holdout — the only expensive item

No asset in the current universe qualifies; all 8 are burned. A clean holdout must be new data.

**Proposal.** Freeze **6 tickers never loaded in this project**, spanning the same asset classes
as the original panel, plus both alpha levels. Never inspect them until the final specification
is frozen, then score **once**.

- Candidate universe (to be fixed by Risk, not by me, to avoid another selection path):
  one broad index, one non-US index, one large-cap equity, one metal, one energy, one FX —
  none overlapping `^GSPC, BTC-USD, TSLA, NVDA, SQM, CLP=X, HG=F, CL=F`.
- Specification frozen **before** the download: linear quantile model only (capacity claims are
  withdrawn), parametric and historical anchors, weight grid `{0, 0.1, 0.25, 0.5, 1}`,
  argmin selection, 10 seeds, monthly refit, existing splits.
- Estimated cost at the measured rate (64 s/VAL fit): **~11 h**, one night.
- **One scoring pass. The ledger from 3.5 enforces it.**

Confirmatory predictions, registered before the run:

1. Selection error on the holdout falls in **[0.15, 0.65]** — the primary set's CI. A value
   outside it falsifies the "selection is noise" claim.
2. Seed-IQR reduction holds in **>= 80%** of active comparisons.
3. **No** anchored spec beats its unanchored ablation after Holm correction.

Prediction 3 is the one that matters: it predicts a null, so a positive result would falsify
the paper's own thesis rather than confirm it.

## 5. What I am NOT proposing

- **No more selection variants.** Four configurations already agree. Searching for a fifth that
  rescues a result is what the charter forbids.
- **No MLP rescue.** Tuning the MLP now, after seeing it lose, would be the clearest
  forking path in the project. Capacity is dropped from the paper's claims entirely; the stage-A
  VAL evidence stays as a documented observation with its untuned-configuration caveat.
- **No alpha = 0.05 stage B.** Deferred permanently; the capacity arm is closed.

## 6. Sequencing

1. Risk signs off on this plan and **fixes the holdout universe** (section 4).
2. Code work 3.1–3.5, with tests. (~1 day)
3. Rewrite results against the primary set, Holm-corrected, with bootstrap intervals.
4. Freeze the final specification. Risk verifies the freeze.
5. Holdout download + single scoring pass. (~11 h)
6. Return to G4 for re-review, then G5.

Step 1 before step 5 is the point: if I choose the holdout universe, F5 happens again.
