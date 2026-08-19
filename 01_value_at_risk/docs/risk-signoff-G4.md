# Risk sign-off on the G4 remediation plan

> **SUPERSEDED — see `docs/falsification-result-N2.md` and `docs/risk-signoff-G4-final.md`.**
> The variance-reduction claim discussed below was later shown to be **tautological**: an
> uninformative prior reduces inter-seed dispersion just as much (Wilcoxon p = 0.844). This
> document is retained unedited as a dated record of what was believed at the time; it is a
> *record*, not a summary, and correcting it would destroy the history the paper reports.



Reviewer: **Quant Risk**. Reviewing `docs/remediation-plan-G4.md` against the veto in
`docs/risk-review-G4.md`. Date: 2026-08-19.

## DECISION: **CONDITIONAL SIGN-OFF.**

The veto is lifted **subject to R1–R4 below being incorporated before any holdout run.**
Sections 2, 3 and 5 of the plan are accepted as written. Section 1 is accepted after
verification. Section 4 is **not** accepted as written.

---

## Accepted — section 1, the first-touch subset

The Lead flagged this himself as a post-hoc data-selection decision and offered a defence.
I tested the defence rather than taking it.

**The test that matters is directional:** if the exclusion had been outcome-motivated, it would
have moved the headline numbers *toward* the paper's thesis. It moves them away from it.

| | stage 1 full | first-touch subset | direction |
|---|---|---|---|
| selection error | 45% | **38%** | further from 50% -> weakens "selection is noise" |
| seed-IQR reduction (median) | 16.1x | **13.5x** | smaller -> weakens "the anchor stabilises" |
| IQR reduction count | 19/20 | 15/16 | comparable |

Both headline claims got **weaker** under the exclusion. Combined with a selection criterion
that is mechanically verifiable from file timestamps and independent of any result, this is an
adequate defence. **Accepted, on condition that this exact comparison table is printed in the
paper** so a referee can run the same check. A post-hoc subset defended only in prose would not
have passed.

## Accepted — sections 2, 3, 5

No objection. Item 3.5 (the TEST-touch ledger) is the single most valuable line in the plan and
should be built **before** anything else; it is the control whose absence produced 1 899
evaluations. Section 5's refusal to tune the MLP after seeing it lose is correct and I endorse
it explicitly.

One addition to 2.5: keeping the stage-A VAL capacity evidence "as a documented observation"
invites the reader to draw the conclusion the paper has withdrawn. Either print the
untuned-configuration caveat **in the same figure caption**, or cut it.

---

## R1 — REJECTED: prediction 1 is not falsifiable

The plan registers "selection error on the holdout falls in [0.15, 0.65]". Tested against every
plausible true error rate at n = 16:

| true error | observed | falls in [0.15, 0.65]? |
|---|---|---|
| 20% | 3/16 | yes |
| 30% | 5/16 | yes |
| 40% | 6/16 | yes |
| 50% | 8/16 | yes |
| 60% | 10/16 | yes |

Only absurd values (10%, 70%+) fall outside. A prediction that survives almost any outcome is
not a prediction. **Replace with:** *the 95% CI of the holdout selection error contains 0.50.*
That is falsified whenever selection is genuinely informative, which is exactly the claim at
stake.

## R2 — REQUIRED: a power analysis, which this project has never done

At the proposed holdout size (~16 active comparisons), power to reject "error = 50%":

| true error | power |
|---|---|
| 20% | 80% |
| 30% | 45% |
| **35%** | **29%** |

**The study cannot distinguish 30% from 50%.** Every null in this project — four selection
configurations, the capacity arm, the MCS non-discrimination — has been reported without anyone
asking what effect size the design could detect. Some of those nulls may be underpowered rather
than true.

Required: state the minimum detectable effect for every null claim in the paper, and either
size the holdout to reach 80% power against a pre-specified effect, or report the nulls
explicitly as "underpowered, cannot exclude an effect of size X".

## R3 — REQUIRED: prediction 3 is near-certain and adds little

"No anchored spec beats its unanchored ablation after Holm" over ~24 comparisons requires
p < 0.002 to fail. On the primary set exactly 1 of 26 cleared that bar. Predicting it will not
happen again is predicting the base rate. Keep it for completeness, but it must not be
presented as the confirmatory test. The confirmatory test is R1.

## R4 — REQUIRED: the holdout universe needs a mechanical rule, not my judgement

The plan hands me the universe to avoid F5 recurring. Correct instinct, insufficient execution:
Risk choosing by judgement reproduces exactly the failure of "spanning the observed behaviours".

Required: a **written mechanical rule fixed before any download**, e.g. by asset class, the
most liquid instrument by average dollar volume over a stated pre-period, excluding the eight
already burned, with ties broken alphabetically. The rule is registered; the resulting universe
is whatever it produces. I will sign the rule, not the tickers.

---

## Conditions of sign-off

1. R1 — replace prediction 1 with the CI-contains-0.50 form.
2. R2 — power analysis for every null; resize the holdout or label the nulls underpowered.
3. R3 — demote prediction 3.
4. R4 — mechanical selection rule for the holdout universe, registered before download.
5. Print the directional-check table from section 1 in the paper.
6. Build the TEST-touch ledger (3.5) first, and route the holdout run through it.

On completion, G4 passes and the project proceeds to G5.

## Statement

The reframed paper is stronger than the one the study set out to write, and the Lead reached
that conclusion without being pushed. The remaining disagreements are about the falsifiability
and power of the confirmatory design, not about the honesty of the analysis. I have no
outstanding concern about the integrity of the work.
