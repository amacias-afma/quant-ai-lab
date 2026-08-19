# 🏛️ AFMA Quant-AI Lab

[![Made with Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Notebooks](https://img.shields.io/badge/Jupyter-Notebooks-orange?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Status: Active R&D](https://img.shields.io/badge/Status-Active%20R%26D-brightgreen)]()

> *"Complexity must earn its place."*

A research-grade quantitative laboratory dedicated to bridging the gap between **classical financial econometrics** and **modern continuous-time deep learning**. Every model built here is benchmarked against rigorous classical baselines — and only retained if it can mathematically and empirically defeat them.

---

## 🎯 Core Philosophy

In quantitative finance, machine learning models routinely overfit to noise and fail catastrophically out-of-sample. This lab operates on a strict principle: **no AI model is accepted unless it can mathematically and empirically defeat a highly-optimized, domain-specific classical baseline.** Every neural architecture is stress-tested against structural market breaks, exogenous shocks, and volatility clustering.

---

## 🧪 Author & Context

This research is conducted by **Álvaro F. Macías, PhD**, as part of an independent R&D agenda that informs my professional practice in quantitative risk and ALM modeling for systemic banks.

- 🎓 PhD in Applied Mathematics — Instituto de Matemática Pura e Aplicada (IMPA), Brazil
- 🏛️ 15+ years in systemic banks: Banco Internacional · BCI · MUFG · BTG Pactual
- ⚖️ Judicial Expert (2026-2027) — four Chilean appellate courts
- 📚 Book chapter (Transactions of ADIA Lab, 2025) · Springer (2015) · Brazilian patent (2016)
- 🔗 [LinkedIn](https://www.linkedin.com/in/alvaro-macias-phd/)  ·  ✉️ [alvaro.f.macias.a@gmail.com](mailto:alvaro.f.macias.a@gmail.com)

---

## 📂 Research Projects

### [Project 01 — Deep VaR](./01_value_at_risk/)
**Focus:** Physics-Informed Neural Networks (PINNs) for single-quantile risk forecasting.

- Explored the transition from historical simulation to deep learning for predicting the 95% and 99% Value-at-Risk.
- Built a strict classical baseline: GARCH(1,1) with Student-t innovations, MLE-fitted.
- **Key takeaway:** predicting a single quantile leaves the portfolio blind to the *shape* of the tail beyond it. This limitation directly motivated Project 02.

### [Project 02 — Density Forecasting & The Parametric Ceiling](./02_density_forecasting/)
**Focus:** Continuous-time Neural Stochastic Differential Equations (Neural SDEs), Path Signatures, and full probability distributions.

- Moves beyond point estimates to forecast the **entire return distribution**.
- **Classical baselines built:** GARCH(1,1) · MLE-fitted Student-t · proprietary factor-normalized **VIX-Scaled Student-t**.
- **AI architecture:** Rough Path Theory (Path Signatures) and Neural SDEs to capture non-linear, idiosyncratic market microstructure.
- **Status:** active R&D · classical baselines complete · neural architecture in implementation.

---

## 🛠️ Tech Stack & Methodologies

| Layer | Tools / Methods |
|-------|-----------------|
| **Mathematics** | Stochastic Calculus · Rough Path Theory · Maximum Likelihood Estimation · Kolmogorov-Smirnov tests · Anderson-Darling tests |
| **Machine Learning** | Neural SDEs · Deep Signature Transforms · Physics-Informed Regularization · PINNs |
| **Engineering** | Python · PyTorch · SciPy (Nelder-Mead optimization) · Pandas / NumPy (vectorized backtesting) |
| **Reproducibility** | Jupyter Notebooks · `requirements.txt` · seeded RNG |

---

## 🚀 Getting Started

```bash
git clone https://github.com/amacias-afma/quant-ai-lab.git
cd quant-ai-lab
pip install -r requirements.txt
jupyter lab
```

Each project is self-contained in its folder — open the notebooks in order (`01_*.ipynb`, `02_*.ipynb`, etc.) to follow the research narrative.

---

## 📄 Citation

If you use any methodology or code from this lab in academic or applied work, please cite as:

```bibtex
@misc{macias2026quantailab,
  author       = {Álvaro F. Macías},
  title        = {{AFMA Quant-AI Lab}},
  year         = {2026},
  howpublished = {\url{https://github.com/amacias-afma/quant-ai-lab}},
  note         = {Independent R&D in quantitative finance and continuous-time deep learning}
}
```

---

## 📜 License

MIT License — see [LICENSE](./LICENSE).

---

## 🗺️ Roadmap

- [x] Project 01 — Deep VaR (PINN baseline + GARCH benchmark)
- [x] Project 02 — Density Forecasting (classical baselines complete)
- [ ] Project 02 — Neural SDE architecture (in progress)
- [ ] Project 03 — Rough Volatility & Hedged Monte Carlo for crypto-asset infrastructure (planned)
- [ ] Working paper on SSRN (Q3 2026)

---

*Developed as part of the AFMA Quant-AI Lab research series. Built in Santiago, Chile.*
