import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from src.models.lstm import lstm_fit, create_sequences_log

import torch
# from arch import arch_model

def backtest(portfolio_value, confidence_level, df_clean, start_date, end_date, regime_name):
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
    lstm_model, _ = lstm_fit(train_data, seq_len)
    
    # 3b. Prepare Test Sequences (The Key Fix)
    # To predict the first test day, we need the last 'seq_len' days of train data.
    # Get the "Context Window"
    lookback_data = train_data.iloc[-seq_len:]
    
    # Combine lookback + test data to generate valid sequences for the test period
    input_for_test = pd.concat([lookback_data, test_data])
    
    # Generate sequences (X_test)
    # Note: create_sequences_log returns Tensors
    X_test, _ = create_sequences_log(input_for_test, seq_length=seq_len)
    
    # 3c. Predict on Test Data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_model.eval()

    with torch.no_grad():
        # A. Re-run model on Training Data to get in-sample residuals
        X_train, _ = create_sequences_log(train_data, seq_length=seq_len)
        pred_log_train = lstm_model(X_train.to(device)).squeeze().cpu().numpy()
        
        # B. Convert to Annualized Volatility
        # (exp(log_var) / 10000)^0.5 * sqrt(252)
        vol_train_annual = np.sqrt(np.exp(pred_log_train) / 10000) * np.sqrt(252)
        
        # C. Align Training Returns
        # LSTM consumes the first 60 days for context, so predictions start at index 60
        train_returns_aligned = train_data.iloc[seq_len:]
        
        # D. Calculate Standardized Residuals (z = r / sigma)
        # We use daily volatility for normalization (vol_annual / sqrt(252))
        vol_train_daily = vol_train_annual / np.sqrt(252)
        train_z_scores = train_returns_aligned.values / vol_train_daily
        
        # E. Find the Empirical 99% Z-score
        # We look at the 1st percentile (losses) and take the absolute value
        # If the data was Normal, this would be ~2.33. For Crypto/Crisis, it might be 3.0+
        empirical_1_percentile = np.percentile(train_z_scores, 1)
        calibrated_z = abs(empirical_1_percentile)
        
        # Safety clamp: Don't let it be lower than Normal (2.33)
        calibrated_z = max(calibrated_z, 2.33)


    with torch.no_grad():
        # Forward pass
        pred_log_var = lstm_model(X_test.to(device)).squeeze().cpu().numpy()
        
    # 3d. Convert predictions back to Annualized Volatility
    # Step 1: Exp() to get Variance (since output is Log-Variance)
    pred_var_scaled = np.exp(pred_log_var)
    
    # Step 2: Un-scale (Divide by 100^2 because input was scaled by 100)
    pred_var = pred_var_scaled / 10000 
    
    # Step 3: Annualize (Sqrt to get Daily Vol -> * Sqrt(252))
    lstm_vol_annual = np.sqrt(pred_var) * np.sqrt(252)
    
    # 3e. Align with Test Index
    # The generated predictions correspond exactly to the test_data length
    lstm_vol_test = pd.Series(lstm_vol_annual, index=test_data.index, name="LSTM_Vol")
    lstm_vol_test = lstm_vol_test.ewm(span=5, adjust=False).mean()
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
        "LSTM_Breach_%": rate_lstm
    }

    # return {
    #     'GARCH': breach_garch.mean(),
    #     'Hist': breach_hist.mean(),
    #     'LSTM': breach_lstm.mean()
    # }



# def backtest(portfolio_value, confidence_level, df_clean, start_date, end_date, regime_name):
# # -----------------------------------------------------
#     # A. Train/Test Split
#     # -----------------------------------------------------
#     # Train: All history BEFORE the crisis
#     train_mask = df_clean.index < start_date
#     # Test: The crisis period
#     test_mask = (df_clean.index >= start_date) & (df_clean.index <= end_date)
    
#     train_data = df_clean.loc[train_mask, 'log_ret'] * 100 # Scale for GARCH
#     test_data = df_clean.loc[test_mask, 'log_ret'] * 100
    
#     if len(test_data) == 0:
#         return None

#     print(f"\n>>> Analyzing Regime: {regime_name}")
#     print(f"    Train Size: {len(train_data)} days | Test Size: {len(test_data)} days")

#     # -----------------------------------------------------
#     # B. Model Training (Fit on Pre-Crisis Data)
#     # -----------------------------------------------------
#     # 1. GARCH(1,1) - Learned parameters from the past
#     am = arch_model(train_data, vol='Garch', p=1, o=0, q=1, dist='Normal')
#     res = am.fit(disp='off')
    
#     # Forecast: We run the "filter" over the test period using fixed Train parameters
#     # This tests if the old parameters hold up in the new regime.
#     # We construct a new model on the FULL data but fix parameters to the TRAIN result.
#     full_data = df_clean.loc[df_clean.index <= end_date, 'log_ret'] * 100
#     am_full = arch_model(full_data, vol='Garch', p=1, o=0, q=1, dist='Normal')
    
#     # 'fix' forces the model to use the training weights
#     # conditional_volatility gives us the series for the whole period
#     res_fixed = am_full.fix(res.params)
    
#     # Slice just the test period forecasts
#     garch_vol_test = res_fixed.conditional_volatility.loc[start_date:end_date]
    
#     # 2. Historical Volatility (22-day Rolling) - The Baseline
#     # Calculates observed volatility of the previous month
#     hist_vol_test = full_data.rolling(22).std().loc[start_date:end_date]

#     # 3. LSTM Volatility 
#     # Calculates observed volatility of the previous month
#     # TODO change to LSTM !!!!!!!!
#     # lstm_daily_vol = comparison_df['LSTM'] / np.sqrt(252)
#     # 1. Prepare Data
#     seq_len = 66
#     # print(train_data)
#     model, lstm_vol_annual = lstm_fit(train_data, seq_len)

#     # Align dates (LSTM loses the first SEQ_LEN days)
#     lstm_dates = train_data.index[seq_len:]
#     lstm_vol_test = pd.Series(lstm_vol_annual, index=lstm_dates, name="LSTM_Vol")

#     # lstm_vol_test = full_data.rolling(252).std().loc[start_date:end_date]

#     # -----------------------------------------------------
#     # C. VaR Evaluation (Did we survive?)
#     # -----------------------------------------------------
#     # Un-scale returns and vol for PnL calc
#     test_ret_unscaled = test_data / 100
#     garch_vol_unscaled = garch_vol_test / 100
#     hist_vol_unscaled = hist_vol_test / 100
#     lstm_vol_unscaled = lstm_vol_test / 100

#     # Calculate Losses (Positive magnitude)
#     daily_loss = -test_ret_unscaled * portfolio_value
#     daily_loss[daily_loss < 0] = 0
    
#     # Calculate Capital Shields (VaR)
#     var_garch = garch_vol_unscaled * confidence_level * portfolio_value
#     var_hist = hist_vol_unscaled * confidence_level * portfolio_value
#     var_lstm = lstm_vol_unscaled * confidence_level * portfolio_value
    
#     # Check Breaches (Loss > Capital)
#     breach_garch = daily_loss > var_garch
#     breach_hist = daily_loss > var_hist
#     print(daily_loss)
#     print(var_lstm)
#     breach_lstm = daily_loss > var_lstm
    
#     rate_garch = breach_garch.mean() * 100
#     rate_hist = breach_hist.mean() * 100
#     rate_lstm = breach_lstm.mean() * 100
    
#     print(f"    GARCH Breach Rate:      {rate_garch:.2f}%  (Target: 1.0%)")
#     print(f"    Historical Breach Rate: {rate_hist:.2f}%")
#     print(f"    LSTM Breach Rate:       {rate_lstm:.2f}%")
    
    
#     # -----------------------------------------------------
#     # D. Visualization (The "Capital Shield")
#     # -----------------------------------------------------
#     plt.figure(figsize=(10, 4))
#     plt.bar(test_data.index, daily_loss, color='gray', alpha=0.3, label='Daily PnL (Loss)')
#     plt.plot(var_garch.index, var_garch, color='red', label='GARCH Shield')
#     plt.plot(var_hist.index, var_hist, color='blue', linestyle='--', label='HistVol Shield')
#     plt.plot(var_lstm.index, var_lstm, color='green', linestyle='--', label='LSTM Shield')
    
#     # Highlight Breaches
#     plt.scatter(var_garch.index[breach_garch], daily_loss[breach_garch], 
#                 color='red', marker='x', s=50, zorder=5)
#     plt.scatter(var_hist.index[breach_hist], daily_loss[breach_hist], 
#                 color='blue', marker='x', s=50, zorder=5)
#     plt.scatter(var_lstm.index[breach_lstm], daily_loss[breach_lstm], 
#                 color='green', marker='x', s=50, zorder=5)
    
#     plt.title(f"Capital Shield Stress Test: {regime_name}")
#     plt.ylabel("Portfolio Value ($)")
#     plt.legend()
#     plt.show()
#     return {
#         "Regime": regime_name,
#         "GARCH_Breach_%": rate_garch,
#         "HistVol_Breach_%": rate_hist,
#         "LSTM_Breach_%": rate_lstm
#     }