# src/models/baseline.py

import pandas as pd
import pandas_gbq
import numpy as np
import os
import sys

# Setup path to find 'src' module
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
    
from arch import arch_model
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
from src.evaluation.backtest import VolatilityBacktester

class GarchBaseline:
    def __init__(self, ticker: str='SPY', data: pd.DataFrame=None):
    # def __init__(self, project_id, ticker='SPY'):
        # self.project_id = project_id
        self.ticker = ticker
        self.data = data
        self.model_fit = None
        self.resid_std = None  # To store Standardized Residuals (Invariants)

    # def load_data(self, source='bigquery'):
    #     """Fetches training data from BigQuery."""
    #     if source == 'bigquery':
    #         query = f"""
    #             SELECT * 
    #             FROM `market_data.{self.ticker}_processed`
    #             ORDER BY date ASC
    #         """
    #         print("Fetching data from BigQuery...")
    #         df = pandas_gbq.read_gbq(query, project_id=self.project_id)
    #         df.set_index('date', inplace=True)
    #         return df
    #     elif source == 'local':
    #         df = pd.read_parquet(f"data/{self.ticker}_processed.parquet")
    #         df.set_index('date', inplace=True)
    #         return df
    #     else:
    #         raise ValueError("Invalid source. Must be 'bigquery' or 'local'.")

    def train(self):
        """Fits GARCH(1,1) and stores standardized residuals."""
        print(f"Training GARCH(1,1) ...")
        
        # Scale returns for numerical stability
        returns_scaled = self.data['log_ret'] * 100
        
        # Define model: Constant Mean, GARCH(1,1) Volatility
        am = arch_model(returns_scaled, vol='Garch', p=1, o=0, q=1, dist='Normal')
        
        # Fit the model
        self.model_fit = am.fit(disp='off')
        
        # Store Standardized Residuals (The "Invariants" in Meucci's terms)
        # These should be White Noise (N(0,1))
        self.resid_std = self.model_fit.std_resid
        return self.model_fit

    def evaluate_fit(self):
        """
        The 'Quant Auditor'. Checks Param Constraints, Significance, and Meucci's i.i.d. props.
        """
        if self.model_fit is None:
            raise ValueError("Model not trained yet.")

        print("\n" + "="*40)
        print("   MODEL DIAGNOSTIC REPORT (MEUCCI CHECK)")
        print("="*40)
        
        # --- CHECK 1: Parameter Constraints (Stationarity) ---
        params = self.model_fit.params
        alpha = params.get('alpha[1]', 0)
        beta = params.get('beta[1]', 0)
        persistence = alpha + beta
        
        evaluation_list = []
        print(f"1. Persistence (alpha + beta): {persistence:.4f}")
        if persistence < 1.0:
            print("   [PASS] Model is Stationary (Mean Reverting).")
        else:
            evaluation_list.append("Explosive")
            print("   [FAIL] Model is Explosive (persistence >= 1). INVALID.")

        # --- CHECK 2: Statistical Significance (P-values) ---
        pvalues = self.model_fit.pvalues
        # Check if alpha and beta are significant (p < 0.05)
        sig_alpha = pvalues.get('alpha[1]', 1.0) < 0.05
        sig_beta = pvalues.get('beta[1]', 1.0) < 0.05
        
        print(f"2. Significance: Alpha p={pvalues.get('alpha[1]'):.3f}, Beta p={pvalues.get('beta[1]'):.3f}")
        if sig_alpha and sig_beta:
            print("   [PASS] Parameters are statistically significant.")
        else:
            evaluation_list.append("Overfitted")
            print("   [WARNING] Some parameters are not significant. Model may be overfitted.")

        # --- CHECK 3: Independence (Ljung-Box & Lag-1 Correlation) ---
        # "Is the present correlated with the past?"
        # We test the SQUARED residuals for remaining volatility clustering
        lb_test = acorr_ljungbox(self.resid_std**2, lags=[10], return_df=True)
        lb_pvalue = lb_test['lb_pvalue'].values[0]
        
        # Simple Lag-1 Autocorrelation
        lag1_corr = self.resid_std.autocorr(lag=1)
        
        print(f"3. Independence (i.i.): Lag-1 Corr = {lag1_corr:.4f}, Ljung-Box p = {lb_pvalue:.4f}")
        if abs(lag1_corr) < 0.05 and lb_pvalue > 0.05:
            print("   [PASS] Residuals look independent (White Noise).")
        else:
            evaluation_list.append("Residuals Autocorrelated")
            print("   [FAIL] Residuals still have autocorrelation. GARCH failed to capture all dynamics.")

        # --- CHECK 4: Identical Distribution (Meucci Stability Test) ---
        # "Does the distribution change over time?"
        # Split data in half and compare CDFs using Kolmogorov-Smirnov Test
        n = len(self.resid_std)
        half1 = self.resid_std[:n//2]
        half2 = self.resid_std[n//2:]
        
        ks_stat, ks_pvalue = stats.ks_2samp(half1, half2)
        
        print(f"4. Stability (i.d.): KS Test p-value = {ks_pvalue:.4f}")
        if ks_pvalue > 0.05:
            # We cannot reject null hypothesis that they are same distribution
            print("   [PASS] Distribution is stable over time (Regimes are consistent).")
        else:
            evaluation_list.append("Residuals Distribution Changed")
            print("   [FAIL] Distribution changed in second half. Structural Break detected.")

        print("="*40 + "\n")
        if len(evaluation_list) == 0:
            model_pass = True
        else:
            model_pass = False
        return model_pass, evaluation_list
    
    def run_backtest(self, data_test):
        """
        Evaluates on 'Test'.
        """
        
        # 1. Forecast the FUTURE (Test Set)
        # Ideally, we re-fit every day. For speed, we do a "Rolling Forecast" 
        # using the fixed parameters from the training set but updating the history.
        
        print("Generating out-of-sample forecasts...")
        scale = 100
        
        # We need the full dataset to allow the GARCH filter to run through
        full_returns = data_test['log_ret'] * scale

        # Optimized approach using library features:
        # We re-specify the model on the FULL dataset but fix the parameters to the TRAIN values.
        am_full = arch_model(full_returns, vol='Garch', p=1, o=0, q=1, dist='Normal')
        res_full = am_full.fix(self.model_fit.params)
        
        # The .conditional_volatility attribute gives the one-step-ahead sigma
        # for every point in the series based on previous data.
        all_forecasts = res_full.conditional_volatility
        all_variance = res_full.conditional_volatility**2

        # Get the variance forecast
        pred_var_scaled = all_variance.iloc[split_idx:]
        
        # Rescale: Variance scales by 100^2 = 10000
        pred_var = pred_var_scaled / (scale**2)
        
        # Get the Target (Next Day Variance from Ingest)
        # We need to be careful: df['next_day_variance'] at index T is r_{T+1}^2
        # Arch conditional_vol[T] is estimate of r_T^2. 
        # So we need to compare pred_var[T+1] vs target[T]?
        # EASIER: Just grab the 'next_day_variance' column corresponding to the same dates.
        
        # Garch Estimate for Today vs Reality for Today.
        y_pred_var = pred_var
        y_true_var = test_df['target_variance'] # Current day squared return
        
        # Drop NaNs
        valid = ~np.isnan(y_true_var) & ~np.isnan(y_pred_var)
        y_true_var = y_true_var[valid]
        y_pred_var = y_pred_var[valid]

        # Calculate Annualized Vol just for the Plot/RMSE display
        y_pred_vol_ann = np.sqrt(y_pred_var) * np.sqrt(252)
        
        backtester = VolatilityBacktester(y_true_var, y_pred_vol_ann, model_name="GARCH(1,1)")
        
        # TRICK: We pass 'Variance' to compute QLIKE correctly, 
        # but 'Vol' is usually better for RMSE interpretation.
        # Let's stick to passing Variance to the backtester 
        # and letting the backtester handle the conversion logic if you prefer.
        # For now, let's keep it simple: Pass Volatility to the class we wrote earlier.
        
        # # Convert True Variance to True Vol (Annualized)
        # y_true_vol_ann = np.sqrt(y_true_var) * np.sqrt(252)
        
        # # Re-Instantiate Backtester with Annualized Vol
        # backtester = VolatilityBacktester(y_true_vol_ann, y_pred_vol_ann, model_name="GARCH(1,1)")
        
        # Note: Our QLIKE function inside backtester expects Vol inputs and squares them.
        # So passing Vol is correct for the code we wrote previously.
        
        metrics = backtester.compute_metrics()
        backtester.plot_results()
        
        return metrics

    # def run_backtest(self, split_ratio=0.8):
    #     """
    #     Splits data, trains on 'Train', and evaluates on 'Test'.
    #     """
    #     # 1. Split Data
    #     n = len(self.data)
    #     split_idx = int(n * split_ratio)
        
    #     train_df = self.data.iloc[:split_idx]
    #     test_df = self.data.iloc[split_idx:]
        
    #     print(f"\n--- Running Rolling Backtest ---")
    #     print(f"Train Set: {len(train_df)} days | Test Set: {len(test_df)} days")
        
    #     # 2. Train on HISTORY (Train Set)
    #     self.train(train_df)
        
    #     # 3. Forecast the FUTURE (Test Set)
    #     # Ideally, we re-fit every day. For speed, we do a "Rolling Forecast" 
    #     # using the fixed parameters from the training set but updating the history.
        
    #     print("Generating out-of-sample forecasts...")
    #     scale = 100
        
    #     # We need the full dataset to allow the GARCH filter to run through
    #     full_returns = self.data['log_ret'] * scale

    #     # Optimized approach using library features:
    #     # We re-specify the model on the FULL dataset but fix the parameters to the TRAIN values.
    #     am_full = arch_model(full_returns, vol='Garch', p=1, o=0, q=1, dist='Normal')
    #     res_full = am_full.fix(self.model_fit.params)
        
    #     # The .conditional_volatility attribute gives the one-step-ahead sigma
    #     # for every point in the series based on previous data.
    #     all_forecasts = res_full.conditional_volatility
    #     all_variance = res_full.conditional_volatility**2

    #     # Get the variance forecast
    #     pred_var_scaled = all_variance.iloc[split_idx:]
        
    #     # Rescale: Variance scales by 100^2 = 10000
    #     pred_var = pred_var_scaled / (scale**2)
        
    #     # Get the Target (Next Day Variance from Ingest)
    #     # We need to be careful: df['next_day_variance'] at index T is r_{T+1}^2
    #     # Arch conditional_vol[T] is estimate of r_T^2. 
    #     # So we need to compare pred_var[T+1] vs target[T]?
    #     # EASIER: Just grab the 'next_day_variance' column corresponding to the same dates.
        
    #     # Garch Estimate for Today vs Reality for Today.
    #     y_pred_var = pred_var
    #     y_true_var = test_df['target_variance'] # Current day squared return
        
    #     # Drop NaNs
    #     valid = ~np.isnan(y_true_var) & ~np.isnan(y_pred_var)
    #     y_true_var = y_true_var[valid]
    #     y_pred_var = y_pred_var[valid]

    #     # Calculate Annualized Vol just for the Plot/RMSE display
    #     y_pred_vol_ann = np.sqrt(y_pred_var) * np.sqrt(252)
        
    #     backtester = VolatilityBacktester(y_true_var, y_pred_vol_ann, model_name="GARCH(1,1)")
        
    #     # TRICK: We pass 'Variance' to compute QLIKE correctly, 
    #     # but 'Vol' is usually better for RMSE interpretation.
    #     # Let's stick to passing Variance to the backtester 
    #     # and letting the backtester handle the conversion logic if you prefer.
    #     # For now, let's keep it simple: Pass Volatility to the class we wrote earlier.
        
    #     # # Convert True Variance to True Vol (Annualized)
    #     # y_true_vol_ann = np.sqrt(y_true_var) * np.sqrt(252)
        
    #     # # Re-Instantiate Backtester with Annualized Vol
    #     # backtester = VolatilityBacktester(y_true_vol_ann, y_pred_vol_ann, model_name="GARCH(1,1)")
        
    #     # Note: Our QLIKE function inside backtester expects Vol inputs and squares them.
    #     # So passing Vol is correct for the code we wrote previously.
        
    #     metrics = backtester.compute_metrics()
    #     backtester.plot_results()
        
    #     return metrics

    # def diagnose(self):
    #     """Checks for stationarity of residuals."""
    #     # The residuals should look like White Noise (no pattern)
    #     # If they still have patterns, GARCH failed to capture the physics.
    #     self.model_fit.plot()
    #     plt.show()

    def forecast_volatility(self, horizon=1):
        """Predicts the next day's volatility."""
        forecasts = self.model_fit.forecast(horizon=horizon)
        
        # Extract variance forecast and convert back to decimal (un-scale)
        next_day_variance = forecasts.variance.iloc[-1].values[0]
        next_day_vol = np.sqrt(next_day_variance) / 100
        
        # Annualize it (sqrt(252))
        annualized_vol = next_day_vol * np.sqrt(252)
        
        print(f"\n--- GARCH FORECAST ---")
        print(f"Next Day Annualized Volatility: {annualized_vol:.4f} ({(annualized_vol*100):.2f}%)")
        return annualized_vol

if __name__ == "__main__":
    # Get Project ID
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "quant-ai-lab")
    
    # Run Workflow
    garch = GarchBaseline(project_id=PROJECT_ID)
    data = garch.load_data()
    
    # Train
    print(data.head())

    garch.train(data[['log_ret']])
    # Check Math
    model_pass, evaluation_list = garch.evaluate_fit()
    print(model_pass)
    print(evaluation_list)
    # Predict
    garch.forecast_volatility()
    garch.run_backtest(data)
    