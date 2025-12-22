import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
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

def lstm_fit(df_clean, seq_len):
    # Data Prep
    X_train, y_train = create_sequences_log(df_clean, seq_length=seq_len)

    # Initialize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VolatilityLSTM(input_size=2).to(device)
    criterion = Log_QLIKE_Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # Train
    print(f"Training LSTM on {device}...")
    model.train()
    epochs = 150 # 150 is good for log-space
    
    for epoch in range(epochs):
        X_batch = X_train.to(device)
        y_batch = y_train.to(device)
        
        optimizer.zero_grad()
        pred_log_var = model(X_batch).squeeze() # Output is Log-Variance
        
        loss = criterion(pred_log_var, y_batch)
        loss.backward()
        optimizer.step()
        
    # In-Sample Prediction (Optional, for checking fit)
    model.eval()
    with torch.no_grad():
        pred_log_var = model(X_train.to(device)).squeeze().cpu().numpy()

    # --- CORRECTION START ---
    # 1. Exponential to invert Log
    pred_var_scaled = np.exp(pred_log_var)
    
    # 2. Un-scale (return was * 100, so variance is * 10000)
    pred_var_lstm = pred_var_scaled / 10000
    
    # 3. Annualize
    lstm_vol_annual = np.sqrt(pred_var_lstm) * np.sqrt(252)
    # --- CORRECTION END ---

    return model, lstm_vol_annual

# def lstm_fit(df_clean, seq_len):
#     # Use the full cleaned dataset for this demonstration
#     # X_train, y_train = create_sequences_log(df_clean['log_ret'], seq_length=seq_len)
#     X_train, y_train = create_sequences_log(df_clean, seq_length=seq_len)


#     # 2. Initialize Model
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = VolatilityLSTM(input_size=2).to(device)
#     criterion = Log_QLIKE_Loss()
#     optimizer = optim.Adam(model.parameters(), lr=0.005)

#     # 3. Training Loop
#     print(f"Training LSTM on {device}...")
#     model.train()
#     epochs = 150
#     loss_history = []

#     for epoch in range(epochs):
#         X_batch = X_train.to(device)
#         y_batch = y_train.to(device)
        
#         optimizer.zero_grad()
        
#         # Forward Pass
#         pred_variance = model(X_batch).squeeze()
        
#         # Calculate Loss (QLIKE)
#         loss = criterion(pred_variance, y_batch)
        
#         # Backward Pass
#         loss.backward()
#         optimizer.step()
        
#         loss_history.append(loss.item())
#         if (epoch+1) % 10 == 0:
#             print(f"Epoch {epoch+1}/{epochs} | QLIKE Loss: {loss.item():.4f}")

#     # Plot Loss
#     plt.figure(figsize=(8, 4))
#     plt.plot(loss_history)
#     plt.title("LSTM Training convergence (QLIKE Loss)")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.show()
#     # 4. Generate Predictions
#     model.eval()
#     with torch.no_grad():
#         pred_var_scaled = model(X_train.to(device)).squeeze().cpu().numpy()

#     # 5. Invert Scaling
#     pred_var = pred_var_scaled / 100# Un-scale predictions: (Value / 100)^2 -> Value / 10000
#     pred_var_lstm = pred_var_scaled / 10000
#     # Convert to Annualized Volatility
#     lstm_vol_annual = np.sqrt(pred_var_lstm) * np.sqrt(252)

#     return model, lstm_vol_annual
