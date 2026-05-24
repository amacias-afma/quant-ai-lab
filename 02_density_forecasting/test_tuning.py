import pandas as pd
from src.data.synthetic import generate_student_t_garch
from src.features.features import create_features
from src.evaluation.tuning import hyperparameter_search, evaluate_model_config
import os
import sys

def test():
    # 1. Generate small synthetic dataset
    print("Generating synthetic data...")
    df_prices, df_vix = generate_student_t_garch(n_days=800, nu=5.0)
    
    # 2. Create features
    window_size = 22 * 3
    df_features = create_features(df_prices, df_vix, window_size, n_lags=1)
    df_features.dropna(inplace=True)
    df_features = df_features.astype('float32')
    
    df_y = df_features['returns']
    df_X = df_features.drop(columns=['returns'])
    
    print("Evaluating Baselines...")
    config_n = {'model_class': 'Historical_Normal'}
    crps_n, ll_n, ks_n = evaluate_model_config(df_X, df_y, config_n, test_window=22, porcentage_train=0.8)
    print(f"Historical Normal   -> CRPS: {crps_n:.4f}, KS Fail: {ks_n:.1%}")
    
    config_t = {'model_class': 'Historical_StudentT'}
    crps_t, ll_t, ks_t = evaluate_model_config(df_X, df_y, config_t, test_window=22, porcentage_train=0.8)
    print(f"Historical StudentT -> CRPS: {crps_t:.4f}, KS Fail: {ks_t:.1%}")

    print("Running Optuna Hyperparameter Search...")
    df_res = hyperparameter_search(df_X, df_y, n_trials=2, test_window=22, porcentage_train=0.8)
    
    print(df_res)
    print("Success!")

if __name__ == "__main__":
    test()
