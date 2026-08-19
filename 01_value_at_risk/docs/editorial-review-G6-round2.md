# Editorial review — G6, round 2

Reviewer: **Editor**. Reviewing draft v1 after E1–E6 were addressed.

## DECISION: **MINOR REVISION**, conditional on E7 below. E4 remains open.

E1, E2, E3, E5 and E6 are satisfied — several beyond what I asked. But this read surfaced a new
problem in the section the paper is proudest of, and it is the same problem the paper exists to
criticise.

---

## E7 — NEW, REQUIRED. §3.5 cherry-picks its own showcase number

The abstract and §3.5 both headline that the worthless anchor **"stabilises 2.5× more than the
perfect one."** That figure comes from **one weight**. Across the grid:

| weight | truth | nonsense | nonsense ÷ truth |
|---|---|---|---|
| 0.001 | 0.97× | 0.97× | 1.00 |
| 0.005 | 1.01× | 1.12× | 1.11 |
| 0.020 | 1.84× | 2.33× | 1.27 |
| **0.050** | 5.57× | 13.85× | **2.49** |

The 2.5× is the largest of four values, and the paper quotes it as the result. Two paragraphs
later it correctly reports the interval-based version — 1.4× (95% CI 1.0–5.6) against 1.7×
(95% CI 1.0–13.9), overlapping — which says the two are **indistinguishable**, not that one is
2.5× the other.

**A paper whose §3.1 is about multiplicity and whose §3.4 is about a cherry-picked aggregate
cannot headline the most extreme cell of its own demonstration.** A referee will quote this
back, and the paper's credibility is the entire product.

To be clear about what is and is not damaged: the demonstration's *core* claim survives
untouched — the uninformative anchor stabilises **at least as much** as the truth at every
weight, which is all the argument requires. It is the *magnitude* that is cherry-picked.

**Required fix, in order of preference:**

1. **Strengthen the demonstration.** n = 4 non-zero weights and 20 seeds is thin, and the
   bootstrap interval at n = 4 is nearly uninformative (it saturates at the observed range —
   the authors' own test suite documents this). The demonstration is pure numpy and runs in
   seconds: extend to ~10 weights and more seeds and report a real interval. **This is free and
   removes the objection entirely.**
2. Failing that, replace "2.5× more" in the abstract and §3.5 with the interval-based
   statement, and report the full weight grid rather than one row.

Do not simply soften the wording. The table must show all four weights either way.

## E8 — MINOR. "Central point in one table" points at the wrong table

§3.5 says *"This is the paper's central point in one table"* immediately after the single-weight
table. The central point is the **dose–response across weights combined with the loss going the
other way**, which is a different (and better) table. Point the sentence at it.

## Now satisfied

- **E1.** §5 states p = 0.125, applies the paper's own §3.2 standard to itself, and demotes the
  claim to a testable mechanism. This is the passage a referee will respect most.
- **E2.** The synthetic demonstration with the analytical contraction `(1 − 2·lr·w)^T` — which
  visibly contains no reference to the anchor — is exactly what was needed to turn an anecdote
  into a mechanism. The finite-budget condition is correctly identified and justified.
- **E3.** "Four failure modes… we claim instances, not coverage." Accepted.
- **E5.** Ledger moved to an appendix. Accepted.
- **E6.** Intervals throughout, plus an honest note that the per-seed losses were not persisted
  and that the gap was found only by attempting the requirement. Reporting *how* a limitation
  was discovered is better practice than most papers manage.

## Still open

**E4 — related work.** Unchanged since round 1 and now the sole blocker:

- The ML-VaR survey has **one paper read in full**. The premise of §1 — that this literature
  reports improvements without these controls — rests on n = 1. The survey protocol written in
  advance is good practice and I credit it, but a protocol is not evidence.
- **Section H (permutation/placebo lineage) is still empty.** The paper's most transferable
  contribution is a control it presents without ancestry. Fisher's permutation test, label
  shuffling in ML reproducibility, and placebo tests in applied econometrics all belong here.
  Giving the control a lineage makes it *more* credible; leaving it bare invites "they
  reinvented a permutation test and did not know it."

## Verdict

**Minor revision.** Fix E7 (preferably by extending the demonstration — it costs seconds), fix
E8, and close E4. On E7 and E8 I can verify at proof stage. **E4 must be closed before I send
this to referees**; everything else is ready.

The paper is close, and it is now doing something I rarely see: applying its own standard to
its own claims, twice, at its own expense. E7 is the third such opportunity and should be taken
in the same spirit.
