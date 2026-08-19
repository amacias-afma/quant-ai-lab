# Risk review — G4, round 2 (after the R2 power analysis)

Reviewer: **Quant Risk**. This supersedes the "What survives" section of
`docs/risk-review-G4.md` and the conditional sign-off in `docs/risk-signoff-G4.md`.

## DECISION: **sign-off WITHDRAWN.** The project has no surviving empirical finding.

The Lead complied with R2 in full and the resulting analysis is correct. Its consequence is
that both findings I previously said survived do not. I got that wrong and am retracting it.

---

## Retraction 1 — N1 does not survive (my error, corrected by the Lead's own analysis)

In round 1 I wrote that the selection-error result "replicates and does not depend on TEST
being pristine". The power analysis shows the design had **9.5% power** against the observed
effect, an MDE band of [0.15, 0.85], and would need ~125 comparisons for 80% power. A
procedure with a genuinely useful 30% selection error would have been missed 91% of the time.

I should have demanded the power calculation in round 1 rather than endorsing the null. The
Lead also correctly retracted the "four independent replications" framing: those runs share
tickers and periods and are not independent.

**N1 is not evidence that selection is noise. It is evidence that we cannot tell.**

## Retraction 2 — N2 is arithmetic, not a finding

This is the failure I should have caught first, and did not.

The anchor is an L2 penalty pulling the forecast toward a **fixed** prior — fixed across seeds.
As the weight rises, every seed is dragged toward the same point, so inter-seed dispersion must
fall. In the limit w -> infinity all seeds coincide and the IQR is zero **by construction**.
"Anchoring reduces seed dispersion" is therefore a property of shrinkage, not a property of
*this* anchor.

Pooled across all four runs (n = 85 active comparisons), the IQR ratio tracks the selected
weight exactly as pure shrinkage predicts:

| chosen weight | median IQR ratio | n |
|---|---|---|
| 0.10 | 1.49x | 12 |
| 0.25 | 2.57x | 4 |
| 0.50 | 2.48x | 27 |
| 1.00 | 7.66x | 22 |
| 2.00 | 19.69x | 14 |
| 5.00 | 16.02x | 3 |
| 10.00 | 46.60x | 3 |

**Spearman rho = +0.585, p < 1e-4.**

The 15/16 sign test that R2 certified as "informative" is measuring the mechanical consequence
of a penalty term. It would hold for shrinkage toward *any* fixed target, including a
deliberately worthless one.

### Required falsification test (VAL only, no test-set cost)

Re-run the anchored specification with a **nonsense prior** — a constant unrelated to risk
(e.g. the TRAIN unconditional mean return, or a fixed arbitrary level), holding the weight grid
and everything else identical.

- If seed-IQR reduction of comparable magnitude appears, **N2 is confirmed tautological** and
  every variance claim in the paper must be withdrawn or reframed as "shrinkage shrinks".
- If it does not appear, the reduction is specific to an informative prior and N2 partially
  survives.

Until this test is run, **no claim about the anchor reducing dispersion may appear in the
paper.** This test is cheap, uses validation data only, and I regard it as mandatory.

---

## Where this leaves the project

| claim | status |
|---|---|
| Anchoring improves out-of-sample loss | dead (2/32 uncorrected, 1/26 after Holm, no test set) |
| Selection error is chance | **undetermined** — 9.5% power |
| Anchor reduces seed dispersion | **presumed tautological** pending the nonsense-prior test |
| Capacity hurts accuracy | withdrawn (F2 seed noise, F3 untuned MLP, N3 zero power) |
| MCS separates the ladder | no discriminating power at n = 5 |

**No substantive empirical claim remains.** That is not a reason to stop; it is a reason to stop
pretending the paper is about VaR modelling.

## What I do endorse

The Lead's own recommendation, arrived at before this review: publish as a **methods /
negative-results case study**. The asset is the process record, and it is unusually complete:

- a pre-registration with dated amendments, including ones that weakened the authors' position;
- three implementation defects found and documented, each of which had silently favoured the
  authors' hypothesis (early stopping favouring the anchored model, unequal training rows, a DM
  exception hiding the most informative outcome);
- a reconstructed ledger showing **1 959** test-set evaluations across four passes — 60 more
  than I counted, because two debugging runs appear in no manifest;
- a power analysis showing the headline null was never detectable;
- a tautology check dissolving the one finding that survived everything else.

A paper documenting how a carefully pre-registered quantitative study still produced four
dissolving findings, and which control caught each, is more useful to practitioners than
another anchored-VaR result would have been. It requires no holdout, no further compute, and no
claim the data cannot support.

## Conditions to re-open sign-off

1. Run the nonsense-prior falsification test; report the outcome either way.
2. Restate N1 as undetermined with its power figure attached, everywhere it appears.
3. Drop the confirmatory holdout (R1/R3/R4 become moot) **or** size it against a pre-specified
   effect per the R2 table and accept the compute cost.
4. Reframe the paper as above, or bring me a claim the data can carry.

## Note

The Lead has twice brought me analyses that damaged his own position — the directional check on
the first-touch subset, and the power analysis that destroyed his headline. That is the
behaviour this gate exists to reward, and it is the reason I have no concern about the
integrity of the work, only about what can be claimed from it.
