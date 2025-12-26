import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- A. Custom Loss Function (QLIKE) ---
class Log_QLIKE_Loss(nn.Module):
    """
    Physics-Informed Loss: Penalizes under-prediction of volatility shocks.
    Loss = pred_var + (target_sq_ret / exp(pred_log_var))
    """
    def __init__(self):
        super(Log_QLIKE_Loss, self).__init__()

    def forward(self, pred_log_var, target_sq_ret):
        # pred_log_var is already log(sigma^2)
        # loss = log(sigma^2) + r^2 / sigma^2
        #      = pred + target / exp(pred)
        loss = pred_log_var + (target_sq_ret / torch.exp(pred_log_var))
        return torch.mean(loss)

# --- B. The LSTM Model ---
class VolatilityLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=1):
        super(VolatilityLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)
        # NO ACTIVATION FUNCTION (Output is Log-Variance)

    def forward(self, x):
        out, _ = self.lstm(x) 
        last_step_output = out[:, -1, :]
        log_variance_pred = self.fc(last_step_output)
        return log_variance_pred

# --- C. Data Preparation ---
def create_sequences_log(returns, seq_length=60):
    X, y = [], []
    
    # 1. Scale returns
    r_scaled = (returns * 100).values 
    
    # 2. Log-Squared Returns (The "Gaussianized" Signal)
    # Add small epsilon to avoid log(0)
    epsilon = 1e-6
    r_log_sq = np.log(r_scaled**2 + epsilon)
    
    # Stack: [Return, Log-Squared-Return]
    data_features = np.stack([r_scaled, r_log_sq], axis=1)
    
    for i in range(len(data_features) - seq_length):
        X.append(data_features[i:i+seq_length])
        
        # Target is still Squared Return (for the Loss function to evaluate)
        target_variance = r_scaled[i+seq_length]**2
        y.append(target_variance)
        
    return torch.tensor(np.array(X), dtype=torch.float32), \
           torch.tensor(np.array(y), dtype=torch.float32)
        
def create_sequences(returns, seq_length=60):
    """
    Creates sliding windows with TWO features:
    Feature 1: Scaled Log Return (Direction)
    Feature 2: Scaled Squared Return (Magnitude/Energy)
    """
    X, y = [], []
    
    # 1. Scale returns (standard practice)
    # returns are small (0.01), scale to ~1.0
    r_scaled = (returns * 100).values 
    
    # 2. Create Squared Returns (The Volatility Signal)
    r_squared = r_scaled ** 2
    
    # Stack features: (N, 2)
    # Col 0: Return
    # Col 1: Squared Return
    data_features = np.stack([r_scaled, r_squared], axis=1)
    
    for i in range(len(data_features) - seq_length):
        # Input: Window of PAST features
        # Shape: (seq_length, 2)
        X.append(data_features[i:i+seq_length])
        
        # Target: NEXT day's variance proxy (Squared Return)
        # Note: We predict t using [t-60 ... t-1]
        # The target is the squared return at t
        target_variance = r_squared[i+seq_length]
        y.append(target_variance)
        
    return torch.tensor(np.array(X), dtype=torch.float32), \
           torch.tensor(np.array(y), dtype=torch.float32)

def lstm_fit(train_data, seq_len, test_data):
    # Data Prep
    X_train, y_train = create_sequences_log(train_data, seq_length=seq_len)

    # Initialize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_model   = VolatilityLSTM(input_size=2).to(device)
    criterion = Log_QLIKE_Loss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.005)

    # Train
    print(f"Training LSTM on {device}...")
    lstm_model.train()
    epochs = 150 # 150 is good for log-space
    
    for epoch in range(epochs):
        X_batch = X_train.to(device)
        y_batch = y_train.to(device)
        
        optimizer.zero_grad()
        pred_log_var = lstm_model(X_batch).squeeze() # Output is Log-Variance
        
        loss = criterion(pred_log_var, y_batch)
        loss.backward()
        optimizer.step()
        
    # In-Sample Prediction (Optional, for checking fit)
    lstm_model.eval()
    with torch.no_grad():
        pred_log_var = lstm_model(X_train.to(device)).squeeze().cpu().numpy()

    # --- CORRECTION START ---
    # 1. Exponential to invert Log
    pred_var_scaled = np.exp(pred_log_var)
    
    # 2. Un-scale (return was * 100, so variance is * 10000)
    pred_var_lstm = pred_var_scaled / 10000
    
    # 3. Annualize
    lstm_vol_annual = np.sqrt(pred_var_lstm) * np.sqrt(252)
    # --- CORRECTION END ---

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
    
    return lstm_vol_test, calibrated_z
