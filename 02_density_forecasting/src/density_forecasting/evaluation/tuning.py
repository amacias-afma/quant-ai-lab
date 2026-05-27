import optuna
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from density_forecasting.models.tf_neural_networks import train_expanding_window_model
from density_forecasting.evaluation.metrics import evaluate_forecasts, block_ks_test
from density_forecasting.models.neural_networks import generate_montecarlo
from density_forecasting.models.baselines import rolling_vix_scaled_student_t
from scipy import stats

def evaluate_baseline(
    df_X_linear: pd.DataFrame,
    df_y: pd.Series,
    baseline_type: str = 'StudentT',
    porcentage_train: float = 0.8
) -> Tuple[float, float, float]:
    """
    Evaluates a baseline model.
    """
    initial_train_size = int(len(df_X_linear) * porcentage_train)
    
    # Extract historical parameters from df_X for the out-of-sample period
    df_X_test = df_X_linear.iloc[initial_train_size:]
    actual_returns = df_y.iloc[initial_train_size:]
    
    if baseline_type == 'VIX_Scaled_StudentT':
        # Need to generate predictions for the entire dataset then slice
        # The window used for VIX baseline will be 132 (6 months) as defined in nb 03
        window_size = 132
        predicted_ensembles_full = rolling_vix_scaled_student_t(
            returns=df_y, 
            vix=df_X_linear['VIX'], 
            window=window_size
        )
        predicted_ensembles = predicted_ensembles_full[initial_train_size:]
    else:
        # Map the rolling statistical moments back to distribution parameters
        pred_mu = df_X_test['ret_1_mean'].values
        pred_sigma = df_X_test['ret_1_std'].values
        
        # Excess Kurtosis K = 6 / (nu - 4)  =>  nu = 4 + 6 / K
        kurtosis = np.maximum(df_X_test['ret_1_kurt'].values, 0.1)
        pred_nu = 4.0 + (6.0 / kurtosis)
        
        if baseline_type == 'Normal':
            # For Normal, we just generate normal samples
            predicted_ensembles = np.zeros((len(pred_mu), 1000))
            for i in range(len(pred_mu)):
                predicted_ensembles[i, :] = np.random.normal(pred_mu[i], pred_sigma[i], 1000)
        else:
            # StudentT baseline
            predicted_ensembles = generate_montecarlo(
                out_of_sample_mu=pred_mu,
                out_of_sample_sigma=pred_sigma,
                out_of_sample_nu=pred_nu,
                n_samples=1000
            )
        
    df_eval = evaluate_forecasts(actual_returns, predicted_ensembles)
    crps_mean = df_eval['CRPS'].mean()
    log_likelihood_mean = df_eval['Log_Likelihood'].mean()
    
    df_ks = block_ks_test(df_eval['PIT'], block_size=60, alpha=0.05)
    ks_fail_rate = (df_ks['Status'] == '❌ FAILED').mean() if len(df_ks) > 0 else 1.0
    
    return crps_mean, log_likelihood_mean, ks_fail_rate

def evaluate_model_config(
    df_X_linear: pd.DataFrame,
    df_y: pd.Series,
    config: Dict[str, Any],
    df_X_deep: pd.DataFrame = None,
    test_window: int = 22,
    porcentage_train: float = 0.8
) -> Tuple[float, float, float]:
    """
    Evaluates a specific model configuration using the walk-forward backtest.
    """
    model_class = config.get('model_class', 'Linear')
    
    if model_class in ['Historical_Normal', 'Historical_StudentT', 'VIX_Scaled_StudentT']:
        baseline_type = model_class.replace('Historical_', '')
        return evaluate_baseline(df_X_linear, df_y, baseline_type, porcentage_train)

    # Run the walk-forward backtest for Neural Networks
    dates_test, pred_mu, pred_sigma, pred_nu, _ = train_expanding_window_model(
        df_X_linear=df_X_linear,
        df_y=df_y,
        df_X_deep=df_X_deep,
        model_class=model_class,
        epochs=config.get('epochs', 100),
        test_window=test_window,
        porcentage_train=porcentage_train,
        lr=config.get('lr', 0.01)
    )
    
    # Generate Monte Carlo samples
    predicted_ensembles = generate_montecarlo(
        out_of_sample_mu=pred_mu,
        out_of_sample_sigma=pred_sigma,
        out_of_sample_nu=pred_nu,
        n_samples=1000
    )
    
    # Align the actual returns with the predictions
    idx_start = len(df_y) - len(dates_test)
    actual_returns = df_y.iloc[idx_start:]
    
    # Calculate evaluation metrics
    df_eval = evaluate_forecasts(actual_returns, predicted_ensembles)
    
    crps_mean = df_eval['CRPS'].mean()
    log_likelihood_mean = df_eval['Log_Likelihood'].mean()
    
    # Run block KS test for calibration
    df_ks = block_ks_test(df_eval['PIT'], block_size=60, alpha=0.05)
    ks_fail_rate = (df_ks['Status'] == '❌ FAILED').mean() if len(df_ks) > 0 else 1.0
    
    return crps_mean, log_likelihood_mean, ks_fail_rate

def hyperparameter_search(
    df_X_linear: pd.DataFrame,
    df_y: pd.Series,
    df_X_deep: pd.DataFrame = None,
    n_trials: int = 20,
    test_window: int = 22,
    porcentage_train: float = 0.8
) -> pd.DataFrame:
    """
    Runs an Optuna hyperparameter search over model architectures and hyperparameters.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    results = []
    
    def objective(trial):
        # Sample hyperparameters
        model_class = trial.suggest_categorical('model_class', ['Linear', 'WideAndDeep'])
        lr = trial.suggest_float('lr', 1e-4, 5e-2, log=True)
        epochs = trial.suggest_int('epochs', 50, 150, step=50)
        
        config = {
            'model_class': model_class,
            'lr': lr,
            'epochs': epochs
        }
        
        crps_mean, ll_mean, ks_fail_rate = evaluate_model_config(
            df_X_linear, df_y, config, df_X_deep, test_window, porcentage_train
        )
        
        # Save custom attributes to trial
        trial.set_user_attr("crps", crps_mean)
        trial.set_user_attr("log_likelihood", ll_mean)
        trial.set_user_attr("ks_fail_rate", ks_fail_rate)
        
        # Optuna objective: minimize CRPS, with a penalty if KS fail rate is high
        penalty = 0.0
        if ks_fail_rate > 0.05:
            penalty = ks_fail_rate * 5.0 # Large penalty for failing calibration
            
        return crps_mean + penalty

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    # Extract results into a dataframe
    for trial in study.trials:
        if trial.state.name == 'COMPLETE':
            res = trial.params.copy()
            res['crps'] = trial.user_attrs.get('crps')
            res['log_likelihood'] = trial.user_attrs.get('log_likelihood')
            res['ks_fail_rate'] = trial.user_attrs.get('ks_fail_rate')
            res['objective_val'] = trial.value
            results.append(res)
            
    df_results = pd.DataFrame(results).sort_values(by=['ks_fail_rate', 'crps'])
    return df_results
