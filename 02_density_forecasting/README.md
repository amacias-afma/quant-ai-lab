# AFMA Quant-AI Lab | Project 02: Density Forecasting

> **Chapter 01 — Problem Definition & Baselines**

---

## 🎯 Objective

Move beyond predicting a **single number** (e.g., VaR) to forecasting the **entire probability distribution** of returns.

Most quant models split into two worlds: the *Alpha World* (mean/prediction) and the *Risk World* (tails/loss). Density forecasting unifies both: the mean of the forecast distribution is your alpha signal, and the tails are your risk limits — **one model, one source of truth.**

---

## 💡 Key Idea: From Quantiles to Densities

Instead of outputting a single value like a point forecast or a single quantile, our model outputs the **parameters of a probability distribution** (e.g., μ and σ for a Gaussian, or μ, σ, ν for a Student-t). This allows us to reason probabilistically about future returns.

---

## 📐 Evaluation Framework

We evaluate every model through three complementary lenses:

| Metric | Question | Ideal Value |
|--------|----------|-------------|
| **PIT** (Probability Integral Transform) | Is the model *calibrated*? (honest about uncertainty) | Flat uniform histogram |
| **CRPS** (Continuous Ranked Probability Score) | How close is the *whole distribution* to reality? | As low as possible (0 = perfect) |
| **Log-Likelihood** | How much probability did we assign to what actually happened? | As high (least negative) as possible |

### The Acceptance Criterion

A model earns the right to replace its predecessor **only if** it produces a **more uniform PIT distribution** than the current baseline — measured by a lower KS statistic and a higher p-value.

> *Complexity must earn its place. Every model must beat the one before it.*

---

## 🏗️ Project Structure

```
02_forecasting/
│
├── notebooks/
│   └── 01_problem_definition_and_baselines_v2.ipynb   # Main notebook (Chapter 01)
│
├── src/
│   ├── data/
│   │   └── data_loader.py          # Fetch and prepare asset price data
│   │
│   ├── evaluation/
│   │   ├── metrics.py              # CRPS, Log-Likelihood, PIT, financial risk summary
│   │   └── plotting.py             # PIT histogram, CRPS rolling plot, distribution plot
│   │
│   ├── models/
│   │   └── baselines.py            # Historical Simulation, Gaussian, Student-t rolling baselines
│   │
│   └── features/                   # (reserved for feature engineering in future chapters)
│
├── data/                           # Raw and processed data (gitignored)
├── config/                         # Configuration files
├── tests/                          # Unit tests
└── requirements.txt
```

---

## 📓 Notebook: `01_problem_definition_and_baselines_v2.ipynb`

### Structure

| Section | Description |
|---------|-------------|
| **1. Project Lineage** | Evolution from Deep VaR (Project 01) to full density forecasting |
| **2. Motivation** | Why Alpha and Risk should share one model |
| **3. Problem Definition** | The non-stationarity trap and the "one observation" dilemma |
| **4. Methodology** | The augmented state space (endogenous + exogenous signals) |
| **5. The Benchmarks** | Historical Gaussian and Historical Student-t baselines |
| **6. Success Metrics** | Deep dive into PIT, CRPS, and Log-Likelihood |
| **7. The Construction Framework** | Three steps before training any model |
| **Setup & Imports** | Environment setup |
| **Step 2 in Action** | Synthetic experiments (Perfect World, Oracle, Broken Oracle) |
| **Step 3 in Action** | Real market data — baseline evaluation on AAPL |
| **Results Registry** | Side-by-side comparison table of all baselines |
| **Next Steps** | Roadmap to future notebooks |

### Baselines

| Model | Distribution | Key Property |
|-------|-------------|--------------|
| **Historical Gaussian** | N(μ, σ) — rolling 252-day window | Simple, fast, ignores fat tails |
| **Historical Student-t** | t(μ, σ, ν) — MLE fit, rolling 252-day | Fat-tail aware; industry-standard upgrade |

---

## 🗺️ Roadmap

| Notebook | Focus |
|----------|-------|
| **01** ✅ | Problem definition, evaluation framework, statistical baselines |
| **02** 🔜 | GARCH/EGARCH — classical volatility baseline |
| **03** 🔜 | First Neural Network — Deep Hybrid, trained with log-likelihood |
| **04** 🔜 | Physics-Informed calibration (PINN) — add statistical constraints |
| **05** 🔜 | Augmented financial signals (VIX, bond yields, inflation, DXY) |
| **06** 🔜 | Non-financial signals (sentiment, geopolitical/macro event risk) |

---

## ⚙️ Setup

```bash
# 1. Create and activate the conda environment
conda create -n quant-ai-lab python=3.10
conda activate quant-ai-lab

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the notebook
jupyter notebook notebooks/01_problem_definition_and_baselines_v2.ipynb
```

---

## 🔗 Related Projects

- **Project 01 — Deep VaR:** Physics-informed neural network for single-quantile VaR forecasting (the predecessor to this work)

---

*Part of the AFMA Quant-AI Lab research series.*
