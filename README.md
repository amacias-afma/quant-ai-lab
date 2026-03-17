# AFMA Quant-AI Lab

> **A research series exploring the intersection of Deep Learning and Quantitative Finance.**

The goal is to build models that are not just statistically sound, but practically useful — bridging the gap between academic theory and real-world trading and risk management.

Each chapter tackles a progressively harder problem, and every new model must **earn its complexity** by beating the previous baseline.

---

## 🗺️ Research Roadmap

| # | Project | Status | Core Question |
|---|---------|--------|---------------|
| **01** | [Physics-Informed Deep VaR](#01--physics-informed-deep-value-at-risk) | ✅ Complete | Can a Neural Network beat Historical Simulation for single-quantile VaR? |
| **02** | [Density Forecasting](#02--density-forecasting) | 🔄 In Progress | Can we forecast the *entire* return distribution — unifying Alpha and Risk? |

---

## 01 · Physics-Informed Deep Value at Risk

📁 `01_value_at_risk/`

### The Problem

Estimating the **99% Value at Risk (VaR)** for Bitcoin using Deep Learning. Standard LSTMs failed due to data starvation and vanishing gradients — the "flat line" problem.

### The Solution: Physics-Informed Hybrid AI

Instead of letting the network guess freely, we **anchor it** to a classical statistical prior (Parametric VaR) via a custom hybrid loss function:

$$\mathcal{L} = \text{PinballLoss}(y, \hat{y}) + \lambda \cdot (\hat{y} - \text{Anchor})^2$$

This gives the model the **stability** of classical statistics and the **adaptability** of Deep Learning.

### Key Results

| Model | Breach Rate (Target: 1%) | Capital Efficiency | Responsiveness | Verdict |
|-------|--------------------------|-------------------|----------------|---------|
| Naive LSTM | 0.00% | -10.60% | 0.0072 | ❌ Fails (flat line) |
| Historical Simulation | 1.15% | -6.24% | 0.0109 | ✅ Baseline |
| **Physics-Informed Hybrid** | **0.87%** | **-6.52%** | **0.0093** | 🏆 Winner |

**The Hybrid AI is 2× more responsive** to volatility spikes than Historical Simulation, while maintaining a safer breach rate — paying a tiny capital premium (0.27%) for significantly better calibration.

### Key Takeaways
- **Constraints are King:** In low-signal environments (finance), pure Deep Learning often fails. Anchoring to a statistical prior stabilizes learning.
- **Don't trust the loss:** A lower Quantile Loss can actually mean a *worse* (unsafe) model if it achieves it by being greedy on calm days. Always check the **Breach Rate**.
- **Walk-Forward Validation** is non-negotiable — look-ahead bias silently destroys realistic evaluation.

📄 [Project 01 README](01_value_at_risk/README.md)

---

## 02 · Density Forecasting

📁 `02_density_forecasting/`

### The Problem

Project 01 predicted a **single quantile**. But why stop there? Most quant teams operate in two silos — Alpha (where is the price going?) and Risk (how much can we lose?) — when both answers live inside the same object: the **full return distribution**.

### The Goal: From Quantiles to Densities

Instead of predicting a single VaR number, we estimate the **parameters of a full probability distribution** (e.g., μ and σ for a Gaussian, or μ, σ, ν for a Student-t). This means:

- The **mean** of the forecast distribution → your Alpha signal
- The **tails** of the forecast distribution → your Risk limits
- **One model. One source of truth.**

### Evaluation Framework

Every model is judged by three metrics:

| Metric | Question | Ideal Value |
|--------|----------|-------------|
| **PIT** (Probability Integral Transform) | Is the model *calibrated*? (honest about uncertainty) | Flat uniform histogram |
| **CRPS** (Continuous Ranked Probability Score) | How close is the *whole distribution* to reality? | As low as possible |
| **Log-Likelihood** | How much probability did we assign to what actually happened? | As high (least negative) as possible |

**The Acceptance Criterion:** A model earns the right to replace its predecessor *only if* it produces a more uniform PIT distribution than the current baseline.

> *Complexity must earn its place. Every model must beat the one before it.*

### Chapter 01 — Problem Definition & Baselines (✅ Complete)

Established the evaluation framework and validated it with synthetic experiments. Ran two statistical baselines on real AAPL data:

| Baseline | Distribution | Design |
|----------|-------------|--------|
| **Historical Gaussian** | N(μ, σ) — rolling 252-day window | Simple. Fast. Ignores fat tails. |
| **Historical Student-t** | t(μ, σ, ν) — MLE fit, rolling 252-day | Fat-tail aware. Industry-standard upgrade. |

### Planned Chapters

| # | Notebook | Focus |
|---|----------|-------|
| **01** ✅ | Problem Definition & Baselines | Evaluation framework + statistical baselines |
| **02** 🔜 | GARCH/EGARCH Baseline | Classical volatility baseline |
| **03** 🔜 | First Neural Network | Deep Hybrid architecture, trained with log-likelihood |
| **04** 🔜 | Physics-Informed Calibration (PINN) | Add statistical constraints (fat tails, clustering) |
| **05** 🔜 | Augmented Financial Signals | VIX, bond yields, inflation, DXY |
| **06** 🔜 | Non-Financial Signals | Sentiment, geopolitical and macro event risk |

📄 [Project 02 README](02_forecasting/README.md)

---

## ⚙️ Setup

```bash
# Create and activate the environment
conda create -n quant-ai-lab python=3.10
conda activate quant-ai-lab

# Install dependencies (per project)
pip install -r 01_value_at_risk/requirements.txt
pip install -r 02_forecasting/requirements.txt
```

---

## 🔬 Design Philosophy

1. **Theory-driven, not data-driven alone.** Statistical anchors and domain knowledge are not constraints — they are informative priors that prevent overfitting.
2. **Earn your complexity.** Every new model must beat the previous baseline on a held-out set. Complexity for its own sake is rejected.
3. **Calibration first, sharpness second.** A model that is overconfident is dangerous. We prioritize honesty (PIT uniformity) over narrow predictions.
4. **Narratives matter.** Each project is structured as a story — from the failure of naive approaches to the final, justified solution.

---

*Part of the AFMA Quant-AI Lab research series.*

> ⚖️ **Disclaimer:** This project is for educational and research purposes only. It is not financial advice.
