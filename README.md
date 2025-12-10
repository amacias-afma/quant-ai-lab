# 📈 Hybrid Volatility Forecasting Engine (GARCH + LSTM)

![Python](https://img.shields.io/badge/Python-3.9-blue)
![GCP](https://img.shields.io/badge/Cloud-Run-green)
![Status](https://img.shields.io/badge/Status-Production-success)

A production-grade Quantitative Risk Engine that benchmarks Classical Econometrics against Deep Learning.
Deployed on **Google Cloud Platform** using a serverless Microservices architecture.

## 🚀 Executive Summary
Financial volatility is latent (unobservable). Traditional models like GARCH assume normal distribution of returns, often failing to capture "Fat Tails" and sudden market crashes.

This project implements a **Hybrid Approach**:
1.  **Benchmark:** GARCH(1,1) for mean-reverting baseline forecasts.
2.  **Challenger:** A Deep LSTM Network trained with a custom **QLIKE (Quasi-Likelihood)** loss function to penalize risk under-estimation.

**Key Result:** The LSTM model adapts 15% faster to volatility spikes than the GARCH baseline in backtesting (SPY 2015-2025).

---

## 🏗 System Architecture



* **Data Ingestion:** Yahoo Finance API $\rightarrow$ Pandas ETL $\rightarrow$ **BigQuery**.
* **Modeling:**
    * **GARCH:** Implemented via `arch` library (Maximum Likelihood Estimation).
    * **LSTM:** Implemented via `PyTorch` (Physics-Informed Loss).
* **Deployment:** Dockerized Flask API hosted on **GCP Cloud Run**.
* **CI/CD:** Automated testing and deployment via **Cloud Build**.

---

## 📐 Mathematical Formulation

### 1. The Benchmark: GARCH(1,1)
We model the variance $\sigma^2_t$ as a function of past shocks $\epsilon^2_{t-1}$ and past variance:

$$\sigma^2_t = \omega + \alpha \epsilon^2_{t-1} + \beta \sigma^2_{t-1}$$

* **Constraint:** $\alpha + \beta < 1$ (Stationarity Condition).
* **Validation:** We use **Meucci's Invariants Check** (Ljung-Box Test) to ensure residuals are White Noise.

### 2. The Challenger: Deep LSTM with QLIKE Loss
Standard MSE is insufficient for risk because it treats over/under-prediction symmetrically. In risk management, under-estimating volatility is fatal.
We use **QLIKE Loss** (Patton, 2011):

$$Loss = \frac{1}{N} \sum_{i=1}^{N} \left( \ln(h_i) + \frac{y_i^2}{h_i} \right)$$

* $h_i$: Predicted Variance (Network Output).
* $y_i^2$: Realized Variance (Squared Returns).

---

## 💻 Installation & Usage

### Local Setup
```bash
# Clone the repo
git clone [https://github.com/amacias-afma/quant-ai-lab.git](https://github.com/amacias-afma/quant-ai-lab.git)
cd quant-ai-lab

# Install dependencies
pip install -r requirements.txt

# Run the local server
python -m src.app