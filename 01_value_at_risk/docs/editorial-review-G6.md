# Editorial review — G6, draft v1

Reviewer: **Editor**. Standard applied: would a competent referee at an academic-lite venue
(SSRN / arXiv q-fin, methods track) recommend this without an email fight?

## DECISION: **MAJOR REVISION.** Not desk-rejected. Not submittable as it stands.

The work is honest, the numbers are verified against source, and the artefact is unusual. What
is missing is the difference between an anecdote and a contribution. Six required changes; E1
and E2 are the ones that decide whether this is publishable at all.

---

## E1 — REQUIRED. The paper commits its own central sin, in §5

> *"That all three errors ran the same direction is the paper's most transferable observation."*

Three defects, all favouring the hypothesis. Under a null of independent signs that is
**p = 0.125 one-sided, 0.25 two-sided.** The paper spends §3.2 explaining that a 6/16 result
with p = 0.45 cannot support a claim, then builds its most quotable line on 3/3 with p = 0.125.

A referee will find this in one reading and it will cost the paper its credibility, because
credibility is the *only* thing this paper is selling.

**Fix:** state the count, state the p-value, and demote the claim to an observation that
motivates a hypothesis for others to test. Something like: *three of three ran in the same
direction; with n = 3 this is not evidence (p = 0.125), but the mechanism — a defect that
contradicts your hypothesis gets investigated, one that confirms it gets shipped — is testable
and we suggest it be tested.* The honesty of downgrading your own best line is worth more here
than the line.

## E2 — REQUIRED. The shrinkage result needs a demonstration, not just an instance

The tautology is real but it is **not new**: variance reduction by shrinkage toward a fixed
target is Stein, ridge, and every bias–variance treatment since. As written, contribution 3
claims something a referee will call textbook.

What is potentially new is not the mechanism but that **the mechanism gets reported as an
empirical finding**, and that a cheap control separates the two. The paper asserts this from a
single case.

**Fix — the cheapest high-value work available:** add a **synthetic demonstration** with known
ground truth. Simulate data where the prior is by construction uninformative; show analytically
or numerically that inter-seed IQR falls as `w` grows, reproducing the same dose–response
(ρ ≈ +0.585) that the real study produced. Pure numpy, minutes to run, no torch.

That converts "we were fooled once" into "here is the artefact, isolated, with ground truth,
and here is the control that detects it." It is the difference between a war story and a
method.

**Also required:** cite at least three published studies that report stability-under-
regularisation as a benefit without a scale-matched control. Without evidence that the error
occurs in the literature, the paper's premise is unsupported.

## E3 — REQUIRED. "Taxonomy" is oversold at n = 1

Contribution 2 promises a taxonomy of failure modes. One study yields four *instances*, not a
taxonomy, and §8 already concedes it. Rename to "four failure modes observed in a single
pre-registered study" and drop the claim to completeness. A referee who reads "taxonomy" and
finds n = 1 will distrust the rest.

## E4 — REQUIRED. No related work

There is none. Novelty is unassessable, so the paper is unreviewable. Minimum:

- pre-registration in economics/finance and its adoption rate;
- permutation and placebo controls (this is where the nonsense-prior control belongs — it is a
  variant of a known family, and saying so *strengthens* the paper by giving it lineage);
- multiple testing and the Model Confidence Set in forecast comparison;
- ML-for-VaR results the study set out to check, with their disclosure practices — this is what
  motivates the whole exercise and its absence is conspicuous.

## E5 — Contribution 4 (the ledger) is engineering

Valuable, but it is an artefact, not a research contribution. Move to an appendix or a
software note. Keeping it in the headline list invites "this is a blog post about their repo".

## E6 — Do not submit with open items

The draft ends with unchecked boxes including bootstrap intervals (Risk F7). Every ratio quoted
— 13.5×, 16.1×, 2.91× — appears as a bare point estimate in a paper whose thesis is that people
report numbers without their uncertainty. Close F7 before submission or remove the magnitudes
and report only the sign tests.

---

## Accepted without change

- **§6, the self-audit.** Reporting that the post-hoc subset moves both headline numbers
  *against* the thesis, with the directional table printed, is the strongest passage. Keep it
  exactly as is.
- **§4, disclosure.** 1,959 evaluations, 4 passes, 0 cells scored once, sourced from a ledger
  rather than a hand count. Most papers cannot produce this at all.
- **§7, the cost column.** The observation that the two cheapest controls destroyed the two
  findings everything else passed is the paper's actual argument. Lead with it harder — it
  belongs in the abstract's final sentence, which currently ends on a weaker note.
- **The refusal to soften.** No "suggests", no "tends to". Rare and worth preserving.

## On framing

Title works. The abstract's opening — conceding the VaR result is not the contribution — is
correct and I would keep it against the instinct to bury it.

But the paper currently reads as **confession**, and confession is not a contribution. E2 is
what turns it into one: with a synthetic demonstration and three citations, the claim becomes
*"here is an artefact that appears in this literature, here is why it is inevitable, here is a
28-minute control that detects it, and here is a real study where it fooled a pre-registered
team with p = 5.2e-04."*

That paper I would send to referees. This draft I would not.

## Verdict

**Major revision.** E1 and E2 are mandatory. E3–E6 are required but mechanical.
Re-submit for editorial review after E1, E2 and E4; the rest can be verified at proof stage.
