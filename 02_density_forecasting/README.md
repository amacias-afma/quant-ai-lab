# 📊 Project 02: Density Forecasting & The Parametric Ceiling

> **From Single-Point Estimates to Full Probability Distributions.**

Most quantitative models split into two worlds: the *Alpha World* (mean/prediction) and the *Risk World* (tails/loss). Density forecasting unifies both: the mean of the forecast distribution is your alpha signal, and the tails are your risk limits — **one model, one source of truth.**

## 🎯 The Objective & The Classical Ceiling
The goal of this project is to build an algorithm that accurately forecasts the probability distribution of $T+1$ returns. 

Before introducing Deep Learning, we rigorously established the **Parametric Ceiling**. We built industry-standard classical models (GARCH, Student-t) and subjected them to a strict **60-Day Independent Block K-S Test** across three highly volatile stress-test assets: `ARKK` (Macro-Regime Shifts), `USO` (Exogenous Supply Shocks), and `BTC-USD` (Structural Fat Tails).

### 📉 Empirical Motivation: The Baseline Failure
*Failure Rate by Asset (Percentage of 60-Day Regimes Failed):*

| Asset | Naive Gaussian | Student-t (Fat Tails) | GARCH(1,1) (Volatility Clustering) |
| :--- | :--- | :--- | :--- |
| **ARKK** | 34.6% | 26.9% | 19.2% |
| **USO** | 30.8% | 19.2% | 15.4% |
| **BTC-USD** | 45.0% | 32.5% | 37.5% |

**The Classical Flaw:** These models are strictly *backward-looking*. They rely entirely on historical data, adapting to market crashes only *after* they happen. 

## 🏆 Our Classical Champion: The VIX-Scaled Student-t
To push the parametric equations to their absolute limit, we built a forward-looking hybrid model. By coupling the structural fat tails of the Student-t distribution with the regime-aware, options-implied scaling of the VIX (optimized daily via Nelder-Mead with factor normalization), we drastically reduced the calibration failure rates during crises like the COVID-19 crash.

**The Challenge Ahead:** The VIX model relies on a rigid, linear equation tied to the US equity market. It cannot capture the *idiosyncratic* non-linear geometry of individual assets (like Bitcoin). To solve this, we must transition to AI.

---

## 📐 Evaluation Framework
We evaluate every model (Classical and AI) through three strict lenses:

1. **PIT (Probability Integral Transform):** Is the model *calibrated*? (Measured via the Kolmogorov-Smirnov test).
2. **CRPS (Continuous Ranked Probability Score):** How close is the *whole distribution* to reality?
3. **Log-Likelihood:** How much probability did we assign to what actually happened?

---

## 🗺️ Research Roadmap

| Phase | Status | Focus |
|-------|--------|-------|
| **01** | ✅ | Problem definition, evaluation framework (CRPS/PIT), and naive baselines. |
| **02** | ✅ | **The Parametric Ceiling:** GARCH, VIX-Scaled baselines, and optimizer death-loop resolution. |
| **03** | 🔜 | **AI Transition:** Extracting market geometry using **Path Signatures**. |
| **04** | 🔜 | **Neural SDEs:** Training a continuous-time generator to defeat the VIX-Scaled champion. |

---

## 📚 References & Literature

This project builds upon foundational research in Rough Path Theory, Deep Learning, and Quantitative Finance:

**Path Signatures & Rough Path Theory:**
* Lyons, T. (1998). *Differential equations driven by rough signals*. Revista Matemática Iberoamericana.
* Kidger, P., Bonnier, P., Perez Arribas, I., Salvi, C., & Lyons, T. (2019). *Deep Signature Transforms*. NeurIPS.

**Neural Stochastic Differential Equations:**
* Chen, R. T., et al. (2018). *Neural Ordinary Differential Equations*. NeurIPS.
* Kidger, P., Foster, J., Li, X., & Lyons, T. (2021). *Neural SDEs as Infinite-Dimensional GANs*. ICML.

**Probabilistic Evaluation & Volatility Modeling:**
* Gneiting, T., & Raftery, A. E. (2007). *Strictly Proper Scoring Rules, Prediction, and Estimation*. JASA.
* Bollerslev, T. (1986). *Generalized Autoregressive Conditional Heteroskedasticity*. Journal of Econometrics.