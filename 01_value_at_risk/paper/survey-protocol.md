# Survey protocol — disclosure practices in ML-for-VaR papers

**Written before any paper was read.** The paper's premise is that this literature reports
improvements without the controls whose absence defeated our own study. That premise must be
measured, not asserted — and the coding scheme must be fixed in advance, or we would be
selecting evidence exactly as we accuse others of doing.

## Inclusion criteria (fixed in advance)

1. Forecasts Value-at-Risk (or a return quantile used as VaR) at one or more levels.
2. Uses a machine-learning estimator — neural network, tree ensemble, kernel method — as
   opposed to a purely parametric econometric model.
3. Reports an out-of-sample comparison against at least one benchmark.
4. Peer-reviewed or a working paper with a stable identifier.

**No exclusion on findings.** Papers reporting nulls are included if found; excluding them
would bias the survey toward the very practice we are measuring.

## Extraction schema (fixed in advance)

For each paper, record **verbatim or "not stated"** — never inferred:

| field | question |
|---|---|
| `n_seeds` | How many random initialisations? Is any seed variation reported? |
| `seed_dispersion_reported` | Is a spread across seeds shown (sd, IQR, range), or only a point? |
| `selection_protocol` | Were hyper-parameters chosen on a held-out validation block, on the test set, or unstated? |
| `n_specs_disclosed` | Is the number of specifications/configurations evaluated stated? |
| `multiplicity_correction` | Any correction (Bonferroni, Holm, FDR, MCS, White/Hansen)? |
| `test_reuse_stated` | Any statement of how many times the test period was scored? |
| `power_analysis` | Any minimum detectable effect or power calculation? |
| `stability_claim` | Any claim that the method is more stable/robust/less variable? |
| `stability_control` | If yes: is there a matched control (permutation, placebo, uninformative target)? |
| `consistent_scoring` | Is the loss strictly consistent for a quantile (pinball/FZ0), or a proxy (hit rate, capital)? |
| `coverage_gate` | Kupiec / Christoffersen / conditional coverage reported? |
| `preregistered` | Any pre-registration or pre-analysis plan? |

## What counts as supporting our premise

The premise is supported if, across the sampled papers, **`stability_claim = yes` occurs
together with `stability_control = none`**, and if `n_specs_disclosed`, `test_reuse_stated`
and `power_analysis` are predominantly "not stated".

**The premise is NOT supported** — and the paper must be reframed — if these fields are
routinely reported. We commit to reporting the tally either way.

## Honest limits of this survey, stated in advance

- It is a **convenience sample** of papers reachable and readable in one session, not a
  systematic review. It cannot support a claim about the field's base rate.
- Abstracts and landing pages are often all that is accessible; a field coded "not stated" may
  be stated in a full text we could not read. **Every such case is coded `unread` rather than
  `not stated`.** This distinction is the difference between a finding and an artefact of
  access, and conflating them would be the same error the paper is about.
- We therefore report the survey as **illustrative**, and any quantitative claim about the
  literature must be hedged to the sample actually read.

## Status

Filled in `paper/survey-ml-var.md`. Rows coded `unread` are not evidence.
