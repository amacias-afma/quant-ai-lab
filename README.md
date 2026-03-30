# 🏛️ AFMA Quant-AI Lab

Welcome to the **AFMA Quant-AI Lab**, a quantitative research repository dedicated to bridging the gap between classical financial econometrics and modern continuous-time Deep Learning.

## 🎯 Core Philosophy
> *"Complexity must earn its place."*

In quantitative finance, machine learning models frequently overfit to noise and fail catastrophically out-of-sample. This lab operates on a strict scientific principle: **No AI model is accepted unless it can mathematically and empirically defeat a highly optimized, domain-specific classical baseline.** Every neural architecture built here is stress-tested against structural market breaks, exogenous shocks, and volatility clustering.

---

## 📂 Research Projects

### [Project 01: Value at Risk (Deep VaR)](./01_value_at_risk/)
**Focus:** Physics-Informed Neural Networks (PINNs) for Single-Quantile Risk Forecasting.
* Explored the transition from historical simulation to deep learning for predicting the 95% and 99% Value at Risk (VaR).
* **Key takeaway:** Predicting a single quantile leaves the portfolio blind to the shape of the tail. This limitation motivated the transition to Project 02.

### [Project 02: Density Forecasting & The Parametric Ceiling](./02_density_forecasting/)
**Focus:** Continuous-Time Neural SDEs, Path Signatures, and Full Probability Distributions.
* Moving beyond point estimates to forecast the **entire probability distribution** of tomorrow's returns.
* **Classical Baselines Built:** GARCH(1,1), MLE-fitted Student-t, and a proprietary factor-normalized **VIX-Scaled Student-t** model.
* **AI Architecture:** Implementing Rough Path Theory (Signatures) and Neural Stochastic Differential Equations (SDEs) to capture non-linear, idiosyncratic market microstructure.

---

## 🛠️ Tech Stack & Methodologies
* **Mathematics:** Stochastic Calculus, Rough Path Theory, Maximum Likelihood Estimation (MLE), Kolmogorov-Smirnov Tests.
* **Machine Learning:** Neural SDEs, Deep Signature Transforms, Physics-Informed Regularization.
* **Engineering:** Python, PyTorch, SciPy (Nelder-Mead optimization), Pandas/NumPy (vectorized backtesting).

---
*Developed as part of the AFMA Quant-AI Lab research series.*