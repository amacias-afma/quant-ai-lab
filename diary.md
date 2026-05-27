# Quant AI Lab Diary

This log captures high-level milestones, architectural decisions, and experiments conducted across the Quant AI Lab projects.

---

### May 2026

**1. Modularizing the ML Pipeline (Density Forecasting)**
*   **Action**: Extracted massive notebook cells (`03_01.ipynb`) into clean, reusable Python modules inside `src/`.
*   **Result**: Created `src/models/neural_networks.py` (for models/training loops) and `src/features/features.py` (for feature generation). 
*   **Benefit**: Separated concerns, making code testable and preventing notebook clutter.

**2. Upgrading the TensorFlow Neural Network Models**
*   **Action**: Refactored the `LinearStudentTNet` and `WideAndDeepStudentTNet` in `src/models/tf_neural_networks.py` to correctly take advantage of `@tf.function` compilation.
*   **Action**: Solved a "cold-start" optimization issue during the walk-forward backtest by ensuring the `tf.keras.optimizers.Adam` state variables were preserved between rolling windows. This drastically improved convergence speed on the expanding windows.
*   **Action**: Altered `WideAndDeep` architecture to accept a tuple of `(X_linear, X_deep)`, effectively allowing split pathways for statistical features vs raw features.

**3. Feature Engineering Optimization**
*   **Action**: Replaced the extremely slow `rolling_t_fit` (MLE-based Student-T feature extraction) with fast rolling statistical moments (Mean, Std, Skew, Kurtosis). 
*   **Result**: Greatly sped up `create_features()` in `features.py`. The baseline models were updated to map these moments back into Student-t parameters (e.g., estimating degrees of freedom $\nu$ from rolling kurtosis).

**4. Building the Evaluation Pipeline**
*   **Action**: Created `src/evaluation/tuning.py` and `src/evaluation/metrics.py`.
*   **Result**: Implemented `evaluate_model_config()` to run the standardized walk-forward backtest and automatically evaluate CRPS (sharpness) and the Block K-S Test (calibration). Also added Optuna for hyperparameter search.

**5. The Multi-Ticker Showdown (Parametric vs. ML)**
*   **Action**: Implemented `rolling_vix_scaled_student_t` inside `baselines.py` as an advanced "floor" model that scales volatility dynamically using VIX. 
*   **Action**: Generated `notebooks/05_vix_baseline_vs_ml.ipynb` to compare this advanced baseline against the Wide & Deep neural network across multiple tickers (`ARKK`, `USO`, `BTC-USD`, etc.).
*   **Result**: Created a consolidated Pandas DataFrame table to instantly compare CRPS and Block K-S failure rates.

**6. Web App Architecture & Creation**
*   **Action**: Created `quant-ai-webapp` as a separate repository to cleanly decouple the API and UI from the core data science models.
*   **Action**: Built a FastAPI backend to mock BigQuery integrations, and a Vite (Vanilla JS) frontend with a stunning, premium FinTech light-mode aesthetic (inspired by the AFMA corporate identity).

**7. Lab Standardization & Packaging**
*   **Action**: Restructured the entire lab repository to use standard installable Python packages (`pyproject.toml`) for both `01_value_at_risk` and `02_density_forecasting` (`src/value_at_risk` and `src/density_forecasting`).
*   **Action**: Established a unified `template/` folder containing a boilerplate structure and a `_quarto.yml` configuration.
*   **Result**: Models can now be easily installed and imported into the `quant-ai-webapp` API without duplicating code. Researchers can use Quarto to seamlessly write markdown and generate professional LaTeX PDFs from their notebooks.
