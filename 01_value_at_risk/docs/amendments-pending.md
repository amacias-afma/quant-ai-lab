# Pending amendments to `hypotheses.md` — 01-nn-var

> **SUPERSEDED — see `docs/falsification-result-N2.md` and `docs/risk-signoff-G4-final.md`.**
> The variance-reduction claim discussed below was later shown to be **tautological**: an
> uninformative prior reduces inter-seed dispersion just as much (Wilcoxon p = 0.844). This
> document is retained unedited as a dated record of what was believed at the time; it is a
> *record*, not a summary, and correcting it would destroy the history the paper reports.



> **Copy these blocks verbatim into the `## Amendments` section of
> `quant-articles/projects/01-nn-var/02-research/hypotheses.md`.** They live here only because
> the articles repo was not reachable from the session that wrote them. The pre-registration is
> append-only: paste, do not rewrite anything above it.

---

### 2026-08-16 — H4 seed-noise criterion resolved: report BOTH readings

H4 asks whether the anchored-vs-unanchored gap "exceeds the inter-seed IQR". With two models of
very different seed dispersion that phrase is ambiguous, and on ^GSPC (alpha = 0.05) the two
readings disagree. The criterion is therefore fixed as follows, and **both readings are reported
in every results table** — never one alone.

- **Lenient reading** — `edge_exceeds_anchored_iqr`: the anchored spec's q75 lies below the
  baseline's median. Question answered: *is the model under test reliably better?*
- **Conservative reading** — `edge_exceeds_baseline_iqr`: the gap exceeds the **baseline's**
  inter-seed IQR. Question answered: *is the gap larger than the noise of the comparator?*
- **Verdict** — `seed_noise_verdict` is `detectable` only when both hold, `ambiguous` when they
  disagree, `not detectable` when neither does. **A claim of improvement requires `detectable`
  AND a Diebold-Mariano rejection at 5%.** `ambiguous` is reported as "no established
  difference", not as a win.

Rationale for reporting both rather than choosing: the choice was faced *after* seeing results,
so picking the reading that favours the anchor would be a forking path. Reporting both removes
the degree of freedom. Implemented in `evaluation/harness.compare_to_baseline` and covered by
tests in `tests/test_harness.py`.

No hypothesis is added, removed or reworded. No change to the ladder, splits, or metrics.

---

### 2026-08-19 — H5 stage A result, and stage B scope reduction

**Stage A completed** (5 tickers x 20 seeds, VALIDATION only, alpha = 0.01, zero test-set
evaluations). Gate passed.

*H5a — unanchored relative inter-seed IQR by capacity (median over 5 tickers):*
Linear 0.0042 -> MLP 8x1 0.0497 -> MLP 32x2 0.0740 -> MLP 64x3 0.1479. Monotone in the median,
35x end to end. **The largest model is more seed-unstable than the linear one in 5 of 5
tickers** (2.3x to 39.9x). Per ticker, the full ordering is monotone in only 2 of 5.

*H5b — IQR ratio (unanchored / anchored, w = 0.5), median:*
Linear **0.87** -> MLP 8x1 2.38 -> MLP 32x2 2.53 -> MLP 64x3 **4.06**. At the largest capacity
the anchor reduces seed dispersion in 5 of 5 tickers; at linear capacity in only 2 of 5, and
the median ratio is BELOW 1 — the anchor does not stabilise a model that is already stable.

**Reported form of H5.** The strong "capacity ladder" claim is NOT established: per-ticker
monotonicity holds in 2/5 (H5a) and 1/5 (H5b), so the medians are consistent with a real trend
plus noise but do not license an ordering claim. H5 is therefore reported as the **two-point
comparison, linear vs MLP 64x3**, which is 5/5 in both directions. This weakening is recorded
before stage B is run.

**Interpretive note.** Stage A gives a mechanism for the stage-1/2 nulls: with the linear model
the anchor had nothing to stabilise, which is consistent with VAL disabling it in 12/32
comparisons and with the ~45% weight-selection error. The earlier nulls are not evidence that
anchoring is useless; they are evidence that it was tested where it is not needed. This is an
explanation formed after seeing those results and must be labelled as such, not as a
prediction they confirmed.

**Stage B scope reduction, declared before running.** Measured cost (64 s per VAL fit, observed
in stage A, not estimated) puts the originally registered stage B at ~27 h. Reduced to:
5 tickers (the same ones as stage A), alpha = 0.01, weights {0, 0.5}, architectures
{SimpleQuantileNeuron, QuantileMLP 64x3}, **parametric anchor only** -> ~8 h.

  - Tickers, weights and anchor type all match stage A, so VAL evidence maps onto TEST without
    introducing new assets or hyper-parameters.
  - The historical anchor is dropped because it tied the parametric anchor 8-8 across the three
    earlier runs; the choice is not outcome-driven, but it IS a scope reduction and is
    disclosed as such.
  - alpha = 0.05 and the remaining 3 tickers are deferred, not dropped.

---

### 2026-08-18 — H5: capacity, instability and the anchor as a stabiliser (staged)

**Motivation.** Across three selection configurations (coarse grid + argmin, fine grid +
one-SE, fine grid + argmin) the anchor-weight selection error stayed at 45% / 39% / 37%, none
distinguishable from a coin flip. The one effect that survived all three is variance
reduction: the anchored estimator had lower inter-seed IQR in 19/20, 21/23 and 25/27
comparisons. H5 takes that surviving effect seriously and asks what it is *for*.

**H5.** Seed instability grows with model capacity, and the anchor's stabilising effect grows
with it. Formally, over the capacity ladder
Linear (4 params) → MLP 8x1 (41) → MLP 32x2 (1 217) → MLP 64x3 (8 641), on identical features:

- **H5a (premise).** Unanchored relative inter-seed IQR is increasing in capacity.
- **H5b.** The IQR ratio (unanchored ÷ anchored) is increasing in capacity.
- **H5c.** Out-of-sample pinball of the anchored model relative to the unanchored one improves
  with capacity — i.e. the anchor pays off precisely where the network is unstable.

H5 predicts an **ordering across four capacities**, not a single difference. That is
substantially harder to obtain by chance than a pairwise win and is the reason the hypothesis
is worth testing at all.

**STAGED EXECUTION — a decision gate, written before the numbers are seen.**

*Stage A — validation only, zero test-set cost.* `run_capacity_pilot.py` measures per-seed VAL
pinball across the capacity ladder at weights {0, 0.5}. Seed dispersion is a property of the
fitting procedure and needs no test data, so this stage **consumes no test-set evaluations and
does not increment that disclosure integer**.

  - **Gate:** if H5a fails — bigger models are NOT more seed-unstable — then the anchor has
    nothing to stabilise, H5 is falsified at its premise, and **the full panel is not run**.
    That null is reported as the result.
  - If H5a holds, and H5b shows the ratio growing with capacity, proceed to stage B.

*Stage B — full panel with TEST*, only if stage A passes. Architectures added to the existing
ladder; splits, seeds, metrics, coverage gate and dual H4 reporting unchanged. Test-set
evaluations counted and disclosed as usual.

**Anti-forking-path note.** The gate is directional and fixed in advance: a stage-A failure
stops the study rather than prompting a search for a capacity or weight that rescues it. Stage
A is explicitly a mechanism check, not a result; its numbers are not reported as findings.

---

### 2026-08-17 — STAGE 2: refined weight grid + one-standard-error selection rule

Stage 1 (grid {0, 1, 5, 10}, argmin selection) is complete and reported. This registers a
second, **exploratory** stage. Stage 1 results stand as they are and are NOT superseded;
stage 2 is reported separately and its evaluations are added to the disclosure integers.

**Motivation is from the SELECTION DISTRIBUTION, not from test outcomes.** Across the 32
comparisons the selected weights were: 0 → 12, 1 → 14, 5 → 3, 10 → 3. **81% of selections land
on the two smallest grid points, and the grid has no resolution at all between 0 and 1.** This
is a design defect visible without looking at any TEST number, which is what makes correcting
it legitimate rather than a forking path.

**Change 1 — refined grid.** `{0, 0.1, 0.25, 0.5, 1, 2}`, log-spaced where the selections
concentrate. The upper end is deliberately reduced: only 6 of 32 selections exceeded 1, and
those cases are the ones that transferred worst to TEST.

**Change 2 — one-standard-error selection rule.** The anchored family **nests** the unanchored
model at w = 0, so the anchored spec cannot be worse by construction; it can only lose through
VAL selection error. Measured under plain argmin: **9 of the 20 active comparisons ended worse
than w = 0 on TEST — a 45% selection-error rate**, i.e. the VAL signal for this parameter is
close to noise. The one-SE rule takes the **smallest** weight whose VAL loss lies within one
standard error of the best, so complexity is bought only when the evidence is clear. Available
as `--selection-rule one_se`; implemented in `select_anchor_weight`, covered by tests.

**Pre-registered prediction (falsifiable).** Under the one-SE rule the number of comparisons
where the anchored spec is worse than w = 0 on TEST should fall materially below 9/20. If it
does not, the anchor weight is not learnable from a VAL block of this length, and that is the
finding to report.

**Scope.** Stage 2 runs on the FULL 8-ticker panel at both α levels — not on a subset chosen
because its weights looked interesting, which would reintroduce selection through the back door.

**Unchanged:** hypotheses, ladder, splits, metrics, seed discipline, coverage gate, and the
dual H4 reporting from the 2026-08-16 amendment.

---

### 2026-08-16 — code defects corrected before any reported result

Three implementation defects were found and fixed. All predate any reportable number; the
^GSPC (alpha = 0.05) run produced before the fixes is void and was regenerated.

1. **Early stopping favoured the anchored model.** The rule stopped when two consecutive epochs
   differed by < 1e-6 relative. The pinball objective of a linear model is piecewise linear and
   plateaus between steps, so the unanchored model stopped almost immediately, while the L2
   anchor made the objective smooth and let the anchored model keep training. On synthetic
   traces the anchored variant ran ~42x more epochs per block. The anchor was buying training
   epochs, not information. Replaced with patience-on-best-loss plus a `min_epochs` floor,
   applied identically to every spec.
2. **Specs trained on different rows.** Rows with an undefined anchor prior were dropped only
   when an anchor was active, so `weight = 0` trained on ~252 more rows (the historical prior's
   warm-up) than `weight > 0`. The weight grid was not comparing like with like and the
   unanchored spec received more data than the anchored ones. Fixed by
   `restrict_to_anchor_support`, applied once up front so every rung sees identical rows.
3. **Disclosure integer overstated.** `specifications evaluated` assumed weight selection used
   the reporting seeds; with `--val-seeds` it did not. The count now uses the VAL seed count.
   The ^GSPC run's honest figure is **54**, not the 90 written by the earlier code.

**Grid boundary, reported not fixed.** On ^GSPC the parametric anchor's VAL curve is monotone
decreasing across {0, 1, 5, 10} and selects the endpoint (10), so its VAL optimum lies outside
the grid. The grid is **left unchanged**: widening it after seeing a null would be a forking
path. The boundary hit is disclosed as a limitation, and the corresponding result is reported as
"VAL optimum not interior — selected weight is a grid endpoint".
