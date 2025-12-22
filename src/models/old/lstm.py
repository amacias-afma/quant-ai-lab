# src/models/lstm.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pandas_gbq
import numpy as np
import os
import matplotlib.pyplot as plt
import sys

# Setup path to find 'src' module
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)


# --- 1. THE PHYSICS-INFORMED LOSS (QLIKE) ---
class QLIKE_Loss(nn.Module):
    """
    Quasi-Likelihood Loss for Volatility Forecasting.
    Loss = log(predicted_var) + (target_squared_return / predicted_var)
    """
    def __init__(self):
        super(QLIKE_Loss, self).__init__()

    def forward(self, predicted_var, target_squared_return):
        # Ensure positivity to avoid log(0) or div/0
        epsilon = 1e-6
        pred_var = predicted_var + epsilon
        
        loss = torch.log(pred_var) + (target_squared_return / pred_var)
        return torch.mean(loss)

# --- 2. DATA PREPARATION (Sliding Window) ---
class FinancialTimeSeriesDataset(Dataset):
    def __init__(self, returns, sequence_length=60):
        """
        :param returns: Array of Log Returns
        :param sequence_length: How many past days the LSTM sees (e.g., 60)
        """
        self.sequence_length = sequence_length
        # We need to turn returns into sequences
        self.X = []
        self.y = []
        
        # Create sequences
        # Input: [r_{t-60}, ..., r_{t-1}]
        # Target: r_{t}^2 (The squared return of the NEXT day to predict variance)
        returns_sq = returns ** 2
        
        for i in range(len(returns) - sequence_length):
            seq = returns[i : i+sequence_length]
            target = returns_sq[i+sequence_length] # Next day's realized variance proxy
            
            self.X.append(seq)
            self.y.append(target)
            
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32).unsqueeze(-1) # Add feature dim
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- 3. THE MODEL ARCHITECTURE ---
class VolatilityLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(VolatilityLSTM, self).__init__()
        
        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout
        )
        
        # Fully Connected Layer to map hidden state to Variance
        self.fc = nn.Linear(hidden_size, 1)
        
        # Activation: Softplus (Smooth ReLU) to ensure Variance is ALWAYS POSITIVE
        self.activation = nn.Softplus()

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # We care about the output of the LAST time step
        out, _ = self.lstm(x) 
        
        # Take the last time step output: out[:, -1, :]
        last_step = out[:, -1, :]
        
        # Map to variance space
        variance_pred = self.fc(last_step)
        
        # Enforce positivity
        return self.activation(variance_pred)

# --- 4. THE MANAGER CLASS (Train/Predict) ---
class DeepVolEngine:
    # def __init__(self, ticker: str='SPY', data: pd.DataFrame=None):

    def __init__(self, ticker='SPY', data: pd.DataFrame=None, seq_len: int=60):
        # self.project_id = project_id
        self.ticker = ticker
        self.seq_len = seq_len
        self.data = data
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # def load_data(self):
    #     query = f"""
    #         SELECT date, log_ret, target_variance 
    #         FROM `market_data.{self.ticker}_processed`
    #         ORDER BY date ASC
    #     """
    #     print("Fetching data from BigQuery...")
    #     df = pandas_gbq.read_gbq(query, project_id=self.project_id)
    #     return df

    def train(self, epochs=50, batch_size=32):
        print(f"Training LSTM on {self.device}...")
        
        # Preprocessing: Scale returns? 
        # For volatility, raw log returns are small (0.001). 
        # LSTMs like values around [-1, 1]. Let's Scale by 100.
        scale = 100.0
        returns_scaled = self.data['log_ret'].values * scale
        
        # # 1. Prepare Data
        # # Split Train/Test (80/20) - strictly by time!
        # split_idx = int(len(returns_scaled) * 0.8)
        # train_data = returns_scaled[:split_idx]
        # test_data = returns_scaled[split_idx:]
        
        train_dataset = FinancialTimeSeriesDataset(returns_scaled, self.seq_len)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 2. Init Model
        self.model = VolatilityLSTM().to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = QLIKE_Loss() # <-- THE CUSTOM LOSS
        
        # 3. Training Loop
        loss_history = []
        
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                # Forward
                pred_var = self.model(X_batch).squeeze()
                
                # Loss (Target is squared returns, already computed in Dataset)
                # Note: y_batch is scaled by 100^2. Loss works fine.
                loss = criterion(pred_var, y_batch)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            loss_history.append(avg_loss)
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | QLIKE Loss: {avg_loss:.4f}")

        # Plot Loss
        plt.plot(loss_history)
        plt.title("LSTM Training Loss (QLIKE)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    def predict(self, df):
        """Generates Out-of-Sample predictions for the Backtester."""
        print("Generating LSTM forecasts...")
        self.model.eval()
        scale = 100.0
        
        # We need sequences for the whole test set
        # We take the FULL dataset, create sequences, then slice the Test portion
        returns_scaled = df['log_ret'].values * scale
        
        dataset = FinancialTimeSeriesDataset(returns_scaled, self.seq_len)
        loader = DataLoader(dataset, batch_size=64, shuffle=False) # No shuffle for inference!
        
        all_preds_var = []
        all_targets_sq = []
        
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                pred = self.model(X_batch).squeeze()
                all_preds_var.extend(pred.cpu().numpy())
                all_targets_sq.extend(y_batch.numpy())
                
        # Unscale Variance: Predicted on (Ret*100)^2 -> Divide by 10000
        preds_var = np.array(all_preds_var) / (scale**2)
        
        # The 'dataset' cuts off the first 'seq_len' points.
        # We need to align with the original DF.
        # The predictions correspond to indices [seq_len : end]
        
        # Calculate Annualized Volatility for Output
        preds_vol_ann = np.sqrt(preds_var) * np.sqrt(252)
        
        return preds_vol_ann, np.array(all_targets_sq)/(scale**2)

    def forecast_future(self, n_days=30, n_simulations=100, current_sequence=None):
        """
        Projects volatility n_days into the future using Recursive Monte Carlo.
        
        :param n_days: Number of days to forecast
        :param n_simulations: Number of Monte Carlo paths
        :param current_sequence: The last observed sequence of returns (tensor). 
                                 If None, uses the last sequence from self.data.
        :return: (mean_vol_path, lower_bound, upper_bound) - scaled to Annualized Vol
        """
        print(f"Simulating {n_days} days into future with {n_simulations} paths...")
        self.model.eval()
        scale = 100.0
        
        # 1. Get Initial Sequence
        if current_sequence is None:
            if self.data is None:
                raise ValueError("No data available. Train or load data first.")
            
            # Take last 'seq_len' log returns
            last_returns = self.data['log_ret'].values[-self.seq_len:] * scale
            current_sequence = torch.tensor(last_returns, dtype=torch.float32).view(1, self.seq_len, 1) # (1, seq, 1)
        
        current_sequence = current_sequence.to(self.device)
        
        # Store all simulated volatility paths
        # Shape: (n_simulations, n_days)
        all_vol_paths = np.zeros((n_simulations, n_days))
        
        with torch.no_grad():
            for i in range(n_simulations):
                # Copy the starting sequence for this path
                seq = current_sequence.clone()
                
                for t in range(n_days):
                    # a. Predict Variance for next day
                    pred_var_scaled = self.model(seq).item()  # output is (100*r)^2
                    
                    # Store prediction (convert to Annualized Vol)
                    # pred_var = pred_var_scaled / 10000
                    # vol = sqrt(pred_var) * sqrt(252)
                    # Simplified: sqrt(pred_var_scaled) / 100 * 15.87...
                    vol_ann = (np.sqrt(pred_var_scaled) / scale) * np.sqrt(252)
                    all_vol_paths[i, t] = vol_ann
                    
                    # b. Monte Carlo Step: Sample a return based on predicted variance
                    # We need a return r_{t+1} to feed back in.
                    # r_{t+1} ~ N(0, \sigma_{t+1})
                    # \sigma_{scaled} = sqrt(pred_var_scaled)
                    sigma_scaled = np.sqrt(pred_var_scaled)
                    simulated_return_scaled = np.random.normal(0, sigma_scaled)
                    
                    # c. Update Sequence (Shift window)
                    # seq is (1, 60, 1)
                    # We want to remove first element, add new element at end
                    new_val = torch.tensor([[[simulated_return_scaled]]], dtype=torch.float32).to(self.device)
                    seq = torch.cat((seq[:, 1:, :], new_val), dim=1)
        
        # 3. Aggregation
        mean_forecast = np.mean(all_vol_paths, axis=0)
        p5 = np.percentile(all_vol_paths, 5, axis=0)
        p95 = np.percentile(all_vol_paths, 95, axis=0)
        
        return mean_forecast, p5, p95

if __name__ == "__main__":
    from src.evaluation.backtest import VolatilityBacktester
    
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "quant-ai-lab")
    
    # Init
    engine = DeepVolEngine(project_id=PROJECT_ID)
    df = engine.load_data()
    
    # Train
    engine.train(df)
    
    # Predict (Get full series)
    pred_vol, true_var = engine.predict(df)
    
    # Slice only the Test Set (Last 20%)
    split_idx = int(len(pred_vol) * 0.8)
    
    y_pred_test = pred_vol[split_idx:]
    y_true_test = true_var[split_idx:] # This is Squared Returns (Variance)
    
    # Convert Truth to Vol for Backtester (Standardize Inputs)
    y_true_vol = np.sqrt(y_true_test) * np.sqrt(252)
    
    # Evaluate
    backtester = VolatilityBacktester(y_true_vol, y_pred_test, model_name="Deep-LSTM (QLIKE)")
    backtester.compute_metrics()
    backtester.plot_results()