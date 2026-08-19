# Submission plan

**Decision: TMLR first, JRMV as plan B.** Simultaneous submission is prohibited at both venues,
so "in parallel" means *deciding the order now*, not sending to both. The preprint route is
independent of that order and proceeds immediately.

---

## Why TMLR first

Not because it is more prestigious. Because **its acceptance criterion is the one this paper
spent three months optimising for**:

> "Are the claims made in the submission supported by accurate, convincing and clear evidence?"

and, explicitly:

> work "should not be rejected because it isn't considered *significant* or *impactful*… nor
> based on the method not being *novel enough*."

That neutralises the paper's only real vulnerability. This is a **complete null with no new
method** — precisely the profile that a novelty-and-impact venue rejects on sight and that TMLR
was created to accommodate. Four framings in this paper were revised *downward* against the
literature; a venue that rewards claim-shrinking is the correct home for that.

Secondary reasons:

- **Rolling submission.** No waiting for a cycle. ICML's position track closed 24 Jan 2026 and
  NeurIPS's Evaluations & Datasets track closed 6 May 2026; the next windows are ~Jan and ~May
  2027.
- **Open reviews.** A TMLR rejection produces public, citable referee comments. For this project
  that is a usable artefact, not a loss — it extends the adversarial record the paper is about.
- **The referee we actually want** is one who runs the code. TMLR reviewers are asked to check
  whether evidence supports claims, which is the review this paper is built to survive.

**The risk, stated plainly.** TMLR is an ML venue and this is a VaR study. The mitigation is
structural and already done: §1 leads with the general problem, §3.5 is pure numpy with ground
truth and no market data, and Appendix D shows the failure mode reaching beyond finance. If a
reviewer says "this is a finance paper," the synthetic demonstration is the answer.

## Why JRMV is the right plan B, not a lesser option

The *Journal of Risk Model Validation* lists among its topics, verbatim, **"pitfalls in model
validation techniques"** and **"best practices in model development, deployment, production and
maintenance."** It is a venue whose subject *is* validation, so the methodological framing reads
as native rather than philosophical — the exact objection a mainstream finance journal would
raise.

It reaches the audience §9's practitioner takeaway is written for. Its drawbacks are reach
(subscription, Risk.net) and a smaller academic citation base.

**If TMLR rejects on scope rather than on evidence, go straight to JRMV without re-arguing the
science.** If TMLR rejects on evidence, fix the evidence first — that rejection would be correct.

## Not chosen, and why

| venue | verdict |
|---|---|
| **ICML Position Track** | Wrong *kind* of paper, independent of the closed deadline. The call states that "papers that describe technical research without advocating a position are not responsive." This paper is evidence-first; a position framing would require inflating the claims we just spent months deflating. |
| **NeurIPS Evaluations & Datasets** | Scope fits well — the track welcomes analysis of "failure modes of existing… evaluation practices" and states that "negative results… are welcome." Deadline passed (6 May 2026). **Keep as the 2027 fallback if both journals decline.** |
| **Mainstream finance journals** (JFE, QF, JBF) | Would treat a null result with no new estimator as a desk reject. The paper's contribution is not a VaR finding. |

---

## Preprint: proceed now, both parts

### Zenodo — do this first

Mint a DOI for the code and frozen data. The Reproducibility statement has a placeholder
(`[repository DOI — to be minted at submission]`) that must be filled before either submission.

Include: `src/`, `scripts/`, `tests/`, `data/snapshots/` with manifests, `outputs/*.csv`,
`outputs/test_touch_ledger.jsonl`, `paper/`. The ledger is part of the evidence, not an artefact
of it.

### arXiv — permitted, with one constraint

TMLR **allows preprints at any time**, but review is **double-blind and the submission must be
anonymised**, and the submitted version must not link to a named version. Practical consequence:

- Post to arXiv under **q-fin.RM** (primary), cross-listed **stat.ML**.
- Prepare **two builds from one source**: a named arXiv version, and an anonymised TMLR version
  with author/affiliation stripped, the Zenodo link replaced by an anonymised mirror, and no
  citation of the arXiv preprint as "our earlier work."
- De-anonymisation via arXiv is a known and documented phenomenon. TMLR permits the preprint
  regardless; this is disclosed here so the choice is deliberate rather than accidental.

---

## Pre-submission checklist

**Manuscript**

- [ ] Fill the Zenodo DOI in the Reproducibility statement.
- [ ] Add author, affiliation, contact (named build only).
- [ ] Convert to the venue's format — TMLR supplies a LaTeX style; JRMV has its own.
- [ ] `booktabs` for tables; `\ref{}` for figure cross-references; `microtype`.
- [ ] Regenerate figures at publication DPI; confirm both render in greyscale.
- [ ] Confirm every §/Appendix cross-reference resolves after the format conversion.

**Evidence — run immediately before submitting, not from memory**

- [ ] `python -m pytest -q` — update the count in the Reproducibility statement if it changed.
- [ ] `python scripts/refresh_paper_figures.py --check` — must exit 0.
- [ ] `python -m value_at_risk.evaluation.ledger --summary` — confirm §4's four integers.
- [ ] Re-verify the two numbers most likely to drift: the pooled dose–response
      (ρ = +0.585, n = 85) and the paired sign test (9 of 10, p = 0.021).

> **This checklist exists because of Appendix B.** Both stale-file defects were introduced by a
> correction that regenerated one artefact and not another. The failure mode is not carelessness,
> it is partial regeneration — so the evidence commands are re-run at submission, not trusted.

**Administrative**

- [ ] OpenReview profiles complete for all authors (TMLR requires this, including conflicts and
      publication history).
- [ ] Nominate action editors — prefer someone with reproducibility or evaluation-methodology
      background over a finance background.
- [ ] Funding and competing-interest statements.

---

## Cover-letter argument, in one paragraph

To be adapted per venue, but the claim is the same:

> This paper reports a complete null and no new method. Its contribution is a chain of evidence
> that a statistically convincing finding — pre-registered, multi-seed, ranked on a strictly
> consistent loss, surviving Diebold–Mariano, the Model Confidence Set, a coverage gate,
> replication across four runs and a dose–response at p = 5.2e-04 — was a restatement of the fact
> that shrinkage shrinks, and that a control costing 28 minutes of validation compute detects it
> while no amount of further statistical testing does. We disclose 1,959 test-set evaluations
> across 16 cells with zero cells scored once, reproduce the artefact synthetically with ground
> truth known, and document five citation errors and two stale result files found in our own
> materials. Three of the paper's framings were falsified by its own literature survey and
> revised downward; none was confirmed.
