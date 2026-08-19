# Risk — G4 final disposition

Reviewer: **Quant Risk**. Supersedes `risk-signoff-G4.md` and closes the round-2 review.
Date: 2026-08-19.

## DECISION: **G4 CLOSED.** Proceed to G5 as a methods / negative-results paper.
## No empirical claim about anchored VaR may appear in it.

---

## The falsification test was run and it defeated the last surviving finding

`docs/falsification-result-N2.md`. The scale-matched control — the real prior permuted in time,
identical mean, sd and marginal distribution, correlation with tomorrow's tail ~0.0007 — reduces
inter-seed dispersion **as much as the informative prior**: 3 of 6 cells in the control's
favour, Wilcoxon p = 0.844.

N2 is tautological, as predicted. The variance reduction is a property of pulling every seed
toward a fixed point. Withdrawn in full.

I note that the Lead flagged the aggregate figure (controls 1.32x vs informative 2.36x) as
**misleading and not to be reported**, because it is produced entirely by the `zero` control
shrinking toward an off-scale target. That aggregate would have supported the claim. He
discarded it and reported the matched comparison instead, which destroys it. That is the
correct call and I want it on the record.

## Final status of every claim

| claim | disposition |
|---|---|
| Anchoring improves out-of-sample loss | **dead** — 1/26 after Holm, and no test set exists |
| Anchor-weight selection is chance | **undetermined** — 9.5% power, MDE band [0.15, 0.85] |
| Anchor reduces seed dispersion | **withdrawn** — tautological (falsification test) |
| Higher capacity hurts accuracy | **withdrawn** — three independent grounds |
| MCS separates the ladder | **no discriminating power** at n = 5 |
| H5 capacity ladder | **withdrawn** — measured the same tautology at four model sizes |

**Nothing empirical survives.** The G4 gate closes on that basis, not despite it.

## Conditions carried into G5

1. **`RESULTS.md` is stale and must be rewritten before anything else.** It still headlines
   "the anchor sharply reduces seed variance (median 16x, 19/20)" as a finding. It is the
   reader-facing artefact, **and a copy is already published to Google Drive**. A withdrawn
   claim sitting in a shared folder is the most likely way this error escapes into the world.
   Correct the file and replace the Drive copy.
2. **Dated process documents stay as they are**, with a superseded banner. They are an
   append-only record of what was believed when, and rewriting them would destroy the very
   history that makes the paper worth publishing. Distinguish: *records* are amended by
   banner, *summaries* are corrected.
3. **R1, R3, R4 are moot** — the confirmatory holdout is dropped, per the R2 finding that a
   ~24-comparison holdout reproduces the same blind zone. Should the Lead ever revive it, all
   three conditions reactivate.
4. **R2 satisfied.** Every null now carries its minimum detectable effect.
5. **The ledger is the disclosure source.** 1 959 test-set evaluations, 16 cells, zero cells
   touched only once. The paper quotes that figure from the ledger, not from a hand count.
6. **No claim may be reintroduced by softening.** "Suggests", "tends to", "appears to stabilise"
   are the same claims in weaker grammar. Risk will read the draft for this specifically.

## Assessment

Over this gate the Lead brought me, unprompted, three analyses that damaged his own position:
the directional check on the first-touch subset, the power analysis that reduced his headline
to "undetermined", and the falsification test that destroyed his last finding. Each was
volunteered before I asked for it.

The study found nothing about Value-at-Risk. It produced an unusually complete record of how a
carefully pre-registered quantitative project generates findings that dissolve under
examination — four of them, each defeated by a different, cheap control. That record is the
contribution, and it is a more useful one than the result the project set out to obtain.

**G4 closed. No veto outstanding.**
