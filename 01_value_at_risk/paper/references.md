# References — working bibliography (Editor condition E4)

**Verification status is marked on every entry.** In a paper about research integrity, a
fabricated or misremembered citation would be disqualifying, so nothing here is cited from
memory alone without saying so.

- **[V]** verified this session against a publisher/repository listing
- **[K]** foundational and well established; author/year/venue stated from standing knowledge,
  **still to be checked against the published record before submission**
- **[?]** needs a real literature search — placeholder, do not cite yet

---

## A. Why the paper exists: false discovery in finance

- **[V] Harvey, C. R., Liu, Y. & Zhu, H. (2016).** "…and the Cross-Section of Expected
  Returns." *Review of Financial Studies* 29(1), 5–68.
  *Use:* the anchor citation. 316 published factors; a t-ratio above 3.0 proposed as the new
  hurdle; between 27% and 53% of tested anomalies likely false discoveries. Our §3.1 (3 of 26
  raw rejections becoming 1 under Holm) is a miniature of exactly this.

- **[V] Bailey, D. H. & López de Prado, M. (2014).** "The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting and Non-Normality." *Journal of Portfolio Management*
  (SSRN 2460551).
  *Use:* the closest existing analogue to our disclosure integers. When many variants are tried
  and the best is kept, the maximum statistic is inflated even if every candidate is noise —
  our 1,959 evaluations across four passes are the same failure without the correction.

- **[V] Bailey, D. H., Borwein, J., López de Prado, M. & Zhu, Q. J. (2014).**
  "Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on
  Out-of-Sample Performance." *Notices of the AMS.*
  *Use:* precedent for a methods-first paper in this literature. Useful for framing §7.

- **[K] Ioannidis, J. P. A. (2005).** "Why Most Published Research Findings Are False."
  *PLoS Medicine* 2(8), e124.
  *Use:* one sentence in §1 only. Widely cited; do not lean on it for anything specific.

## B. Pre-registration and the replication crisis

- **[V] Registered Reports in economics** — a practical guide exists in *Journal of the Economic
  Science Association* (2022), and the *Pacific-Basin Finance Journal* has run a replication
  platform since 2019 and has launched a pre-registration initiative.
  *Use:* evidence that pre-registration is arriving in finance but is not yet standard — which
  is what makes our record unusual enough to be worth publishing. **Exact citations still to be
  pulled.**

- **[?] Adoption rates of registered reports by domain** — a *Scientometrics* (2023) article
  assesses rates across research domains. Would let us state quantitatively how rare
  pre-registration is in finance rather than asserting it.

- **[K] Simmons, J., Nelson, L. & Simonsohn, U. (2011).** "False-Positive Psychology."
  *Psychological Science* 22(11), 1359–1366.
  *Use:* researcher degrees of freedom. Directly relevant to our §3.2 and to the four selection
  configurations we ran.

- **[K] Gelman, A. & Loken, E. (2013).** "The Garden of Forking Paths."
  *Use:* our stage 2 / 2b grid-and-rule changes are textbook forking paths, and we say so.

## C. Power — the control that broke our headline

- **[K] Button, K. et al. (2013).** "Power failure: why small sample size undermines the
  reliability of neuroscience." *Nature Reviews Neuroscience* 14, 365–376.
  *Use:* the direct precedent for §3.2. Their argument transfers unchanged: an underpowered
  null is not evidence of absence. Our design had **9.5%** power against its own observed
  effect.

- **[?] Power practice in financial forecasting** — we have found no paper reporting minimum
  detectable effects for VaR backtests. If that gap is real it is worth stating explicitly, but
  it must be searched properly before we claim it.

## D. Forecast evaluation and multiple comparison

- **[K] Diebold, F. X. & Mariano, R. S. (1995).** "Comparing Predictive Accuracy."
  *Journal of Business & Economic Statistics* 13(3), 253–263.
- **[K] Hansen, P. R., Lunde, A. & Nason, J. M. (2011).** "The Model Confidence Set."
  *Econometrica* 79(2), 453–497. *Use:* §3 and the observation that at n = 5 the MCS retains
  all nine models including a demonstrably miscalibrated one.
- **[K] White, H. (2000).** "A Reality Check for Data Snooping." *Econometrica* 68(5),
  1097–1126.
- **[K] Gneiting, T. (2011).** "Making and Evaluating Point Forecasts." *JASA* 106(494),
  746–762. *Use:* why pinball and not "capital reserved" — consistent scoring functions.
- **[K] Fissler, T. & Ziegel, J. F. (2016).** "Higher order elicitability and Osband's
  principle." *Annals of Statistics* 44(4), 1680–1707. *Use:* joint VaR/ES elicitability.

## E. VaR: benchmarks and backtests

- **[K] Engle, R. F. (1982)**, ARCH, *Econometrica* 50(4).
- **[K] Bollerslev, T. (1986)**, GARCH, *Journal of Econometrics* 31(3).
- **[K] Koenker, R. & Bassett, G. (1978).** "Regression Quantiles." *Econometrica* 46(1), 33–50.
- **[K] Kupiec, P. (1995).** "Techniques for Verifying the Accuracy of Risk Measurement Models."
  *Journal of Derivatives* 3(2).
- **[K] Christoffersen, P. (1998).** "Evaluating Interval Forecasts."
  *International Economic Review* 39(4), 841–862.
- **[K] Boudoukh, J., Richardson, M. & Whitelaw, R. (1998).** "The Best of Both Worlds."
  *Risk* 11. *Use:* age-weighted historical simulation, if the HS-window study is folded in.

## F. Machine learning for VaR — the literature we set out to check

**This section is the weakest and must be strengthened before submission.** The paper's premise
is that this literature reports improvements without adequate controls; that premise currently
rests on our impression rather than on a survey.

- **[V] "Quantile convolutional neural networks for Value at Risk forecasting."**
  *Machine Learning with Applications* (ScienceDirect S2666827021000487).
- **[V] "Forecasting stock return distributions around the globe with quantile neural
  networks."** arXiv:2408.07497.
- **[V] Taylor, J. W. (2019).** "Forecasting Value at Risk and Expected Shortfall Using a
  Semiparametric Approach Based on the Asymmetric Laplace Distribution."
  *JBES* 37(1), 121–133.
  > **Correction to our own materials:** our README described this as "forecasting VaR/ES with a
  > quantile-loss neural network." **That is wrong** — Taylor (2019) is a semiparametric
  > asymmetric-Laplace / ES-CAViaR approach, not a neural network. The error is corrected here
  > and in the README. It is a small instance of the paper's own theme and we intend to say so
  > in a footnote.

- **[?] REQUIRED:** three to five ML-VaR papers examined for whether they report seed counts,
  hyper-parameter selection protocol, number of specifications evaluated, and any stability
  claim without a matched control. **Without this the paper's premise is unsupported and
  Editor condition E4 is not met.**

## G. Shrinkage — why the artefact is inevitable

- **[K] Stein, C. (1956)** and **James, W. & Stein, C. (1961).** Inadmissibility of the usual
  estimator; shrinkage dominates in ≥ 3 dimensions.
- **[K] Hoerl, A. E. & Kennard, R. W. (1970).** "Ridge Regression: Biased Estimation for
  Nonorthogonal Problems." *Technometrics* 12(1), 55–67.

*Use:* these establish that variance reduction under shrinkage is classical, which is exactly
why our §3.4 result was never a finding. **Citing them is not a weakness — it is the argument.**
The novelty we claim is not the mechanism but that it is reportable as an empirical result,
survives a pre-registered protocol with p = 5.2e-04 and a dose–response, and is separable by a
cheap control.

## H. Permutation and placebo controls — the control's lineage

- **[?] REQUIRED.** Our nonsense-prior control is a variant of a known family (permutation
  tests; label shuffling as a sanity check in ML; placebo tests in applied econometrics). We
  must cite that lineage. Claiming the control as novel when it is an adaptation would be the
  kind of overreach this paper exists to criticise — and giving it ancestry makes it *more*
  credible, not less.
  Candidates to verify: Fisher's permutation test; label-shuffling sanity checks in the
  ML-reproducibility literature; placebo/falsification tests in difference-in-differences.

---

## Editor's condition E4: status

| requirement | status |
|---|---|
| pre-registration in economics/finance | **partial** — leads found (B), exact citations pending |
| permutation / placebo control lineage | **NOT MET** — section H is empty of verified work |
| multiple testing and MCS in forecasting | **met** (D), pending [K] → [V] verification |
| ML-for-VaR with disclosure practices | **NOT MET** — three verified papers, no survey (F) |

**Two blocking gaps: F and H.** Both need a real literature search, and F needs someone to read
the papers and record what they disclose. Neither can be closed by assertion.
