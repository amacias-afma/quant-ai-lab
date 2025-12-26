import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from src.models.lstm import lstm_fit, create_sequences_log

import torch
# from arch import arch_model

def backtest(portfolio_value, confidence_level, df_clean, start_date, end_date, regime_name, visual=False):
    # -----------------------------------------------------
    # A. Train/Test Split
    # -----------------------------------------------------
    train_mask = df_clean.index < start_date
    test_mask = (df_clean.index >= start_date) & (df_clean.index <= end_date)
    
    # Scale by 100 for numerical stability in models
    train_data = df_clean.loc[train_mask, 'log_ret'] * 100
    test_data = df_clean.loc[test_mask, 'log_ret'] * 100
    
    if len(test_data) == 0:
        return None

    print(f"\n>>> Analyzing Regime: {regime_name}")
    print(f"    Train Size: {len(train_data)} days | Test Size: {len(test_data)} days")

    # -----------------------------------------------------
    # B. Model Training (Fit on Pre-Crisis Data)
    # -----------------------------------------------------
    
    # --- 1. GARCH ---

    am = arch_model(train_data, vol='Garch', p=1, o=0, q=1, dist='Normal')
    res = am.fit(disp='off')
    
    # GARCH Forecast (Filter over test data with fixed params)
    full_data = df_clean.loc[df_clean.index <= end_date, 'log_ret'] * 100
    am_full = arch_model(full_data, vol='Garch', p=1, o=0, q=1, dist='Normal')
    res_fixed = am_full.fix(res.params)
    garch_vol_test = res_fixed.conditional_volatility.loc[start_date:end_date]
    
    # --- 2. Historical Volatility ---
    hist_vol_test = full_data.rolling(22).std().loc[start_date:end_date]

    # --- 3. LSTM Volatility ---
    seq_len = 66
    
    # 3a. Train the model (We ignore the in-sample return value here)
    lstm_vol_test, calibrated_z = lstm_fit(train_data, seq_len, test_data)
    
    # -----------------------------------------------------
    # C. VaR Evaluation
    # -----------------------------------------------------
    # Un-scale returns and vol for PnL calc
    test_ret_unscaled = test_data / 100
    
    # Divide annualized vol by sqrt(252) to get daily vol for VaR
    garch_vol_unscaled = garch_vol_test / 100
    hist_vol_unscaled = hist_vol_test / 100
    lstm_vol_unscaled = lstm_vol_test / 100 / np.sqrt(252) # Already annualized in step 3d

    # lstm_vol_unscaled = lstm_vol_test / np.sqrt(252) # Already annualized in step 3d

    # Calculate Losses (Positive magnitude)
    daily_loss = -test_ret_unscaled * portfolio_value
    daily_loss[daily_loss < 0] = 0
    
    # Calculate Capital Shields (VaR)
    var_garch = garch_vol_unscaled * portfolio_value * confidence_level
    var_hist = hist_vol_unscaled * portfolio_value * confidence_level
    var_lstm = lstm_vol_unscaled * portfolio_value * calibrated_z
    
    # Check Breaches (Loss > Capital)
    # We generally only count breaches where there was an actual loss (>0)
    loss_mask = daily_loss > 0
    
    breach_garch = (daily_loss > var_garch) & loss_mask
    breach_hist = (daily_loss > var_hist) & loss_mask
    breach_lstm = (daily_loss > var_lstm) & loss_mask
    
    print(f"LSTM Breaches: {breach_lstm.sum()} / {len(breach_lstm)}")
    
    rate_garch = breach_garch.mean() * 100
    rate_hist = breach_hist.mean() * 100
    rate_lstm = breach_lstm.mean() * 100
    
    print(f"    GARCH Breach Rate:      {rate_garch:.2f}%  (Target: 1.0%)")
    print(f"    Historical Breach Rate: {rate_hist:.2f}%")
    print(f"    LSTM Breach Rate:       {rate_lstm:.2f}%")
    
    # -----------------------------------------------------
    # D. Visualization (The "Capital Shield")
    # -----------------------------------------------------
    if visual:
        plt.figure(figsize=(10, 4))
        plt.bar(test_data.index, daily_loss, color='gray', alpha=0.3, label='Daily PnL (Loss)')
        plt.plot(var_garch.index, var_garch, color='red', label='GARCH Shield')
        plt.plot(var_hist.index, var_hist, color='blue', linestyle='--', label='HistVol Shield')
        plt.plot(var_lstm.index, var_lstm, color='green', linestyle='--', label='LSTM Shield')
    
        # Highlight Breaches
        plt.scatter(var_garch.index[breach_garch], daily_loss[breach_garch], 
                    color='red', marker='x', s=50, zorder=5)
        plt.scatter(var_hist.index[breach_hist], daily_loss[breach_hist], 
                    color='blue', marker='x', s=50, zorder=5)
        plt.scatter(var_lstm.index[breach_lstm], daily_loss[breach_lstm], 
                    color='green', marker='x', s=50, zorder=5)
        
        plt.title(f"Capital Shield Stress Test: {regime_name}")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.show()

    return {
        "Regime": regime_name,
        "GARCH_Breach_%": rate_garch,
        "HistVol_Breach_%": rate_hist,
        "LSTM_Breach_%": rate_lstm,
        "breach_garch": breach_garch,
        "breach_hist": breach_hist,
        "breach_lstm": breach_lstm,
        "daily_loss": daily_loss,
        "var_garch": var_garch,
        "var_hist": var_hist,
        "var_lstm": var_lstm,
        "test_data": test_data,
        "train_data": train_data,
    }
