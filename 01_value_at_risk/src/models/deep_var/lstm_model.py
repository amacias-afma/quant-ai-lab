
import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np
import random
from src.models.deep_var.parametric_model import priori_value_at_risk

def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 1. The Single Neuron Model
class SimpleQuantileNeuron(nn.Module):
    def __init__(self, input_size):
        super(SimpleQuantileNeuron, self).__init__()
        # A single linear layer: y = wx + b
        self.linear = nn.Linear(input_size, 1)
        
    def forward(self, x):
        return self.linear(x)

# 2. The Math: Quantile Loss Function
class QuantileLoss(nn.Module):
    def __init__(self, alpha=0.05):
        super(QuantileLoss, self).__init__()
        self.alpha = alpha

    def forward(self, preds, target, weight=0, preds_prior=0):
        errors = target - preds
        # The "Pinball" Logic:
        # max( (alpha-1)*error, alpha*error )
        loss = torch.max((self.alpha - 1) * errors, self.alpha * errors) + weight * (preds - preds_prior)** 2
        return torch.mean(loss)

# --- B. The LSTM Model ---
class QuantileLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2):
        super(QuantileLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        
        last_step_output = out[:, -1, :]
        prediction = self.fc(last_step_output)
        return prediction

def train_model(data, 
                model_type='SimpleQuantileNeuron', 
                alpha=0.05, 
                epochs=500, 
                lr=0.01, 
                rolling=22, 
                split_type={'percentage': 0.8}, 
                regularization_pm=None, 
                hidden_size=64,
                num_layers=1,
                pretrained_state_dict=None,
                silent=False,
                seed=42): # <--- ADDED SEED ARGUMENT

    # 1. FIX RANDOMNESS
    set_seed(seed) 
    
    print(f"--- 🧠 Training {model_type} (alpha={alpha}) ---")

    X = data['X']
    y = data['y']
    dates = data['dates']
    
    num_samples = len(X)

    if 'percentage' in split_type:
        split_idx = int(num_samples * split_type['percentage'])
        window_size = num_samples - split_idx
    else:
        date_rep = split_type['date']
        split_idx = len(dates[dates <= pd.to_datetime(date_rep)])
        window_size = 22
    
    test_size = num_samples - split_idx
    iterate_number = test_size // window_size

    # Prepare Anchor Dataframe globally if needed
    if regularization_pm is not None:
        weight = regularization_pm['weight']
        df = regularization_pm['df']
        # Calculate full series of anchors first
        df_value_at_risk = priori_value_at_risk(df, rolling=rolling, alpha=alpha)
        # Ensure alignment: Extract the anchor column as a Tensor/Array aligned with 'dates'
        # This assumes 'dates' covers the same range as df_value_at_risk
        # We need to ensure we can slice this exactly like X and y
        full_anchor_series = df_value_at_risk.loc[dates, 'value_at_risk_param'].values
        full_anchor_tensor = torch.tensor(full_anchor_series, dtype=torch.float32).unsqueeze(1)
        
    X_test_aux = []
    y_test_aux = [] 
    preds_test_aux = []
    dates_test_aux = []
    last_model_state = None

    for i in range(iterate_number):
        # Slicing tensors
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:split_idx + window_size], y[split_idx:split_idx + window_size]
        
        # 2. FIX ANCHOR SLICING
        if regularization_pm is not None:
            # Slice the anchor exactly matching the X_train and X_test indices
            # Note: We must slice from the full tensor we created earlier
            y_priori_train = full_anchor_tensor[:split_idx]
            y_priori_test = full_anchor_tensor[split_idx:split_idx + window_size]
        else:
            weight = 0
            y_priori_train = 0
            y_priori_test = 0

        if split_idx >= len(dates):
            break
        
        dates_test = dates[split_idx:split_idx + window_size]

        if not silent:
            print(f"    Train size: {len(X_train)} | Test size: {len(X_test)}")

        # Init Model
        if model_type == 'SimpleQuantileNeuron':
            model = SimpleQuantileNeuron(input_size=X_train.shape[1])
        elif model_type == 'QuantileLSTM': # Corrected name to match your snippet usage
             # Assuming VolatilityLSTM was renamed or aliased
             # Make sure to import VolatilityLSTM if that is the real name
            model = QuantileLSTM(input_size=X_train.shape[1], hidden_size=hidden_size, num_layers=num_layers)
        
        if pretrained_state_dict is not None:
            model.load_state_dict(pretrained_state_dict)
            if not silent:
                print("--> Loaded pretrained weights (Warm Start)")

        criterion = QuantileLoss(alpha=alpha)
        optimizer = optim.Adam(model.parameters(), lr=lr) 

        history = {"train_loss": [], "test_loss": []}
        loss_prev = np.inf

        # Training Loop
        for epoch in range(epochs):
            
            model.train()
            # If LSTM, input needs to be (Batch, Seq, Feature) -> (Batch, 1, Feature)
            # If SimpleNeuron, input usually (Batch, Feature)
            # Check your model definition. Assuming unsqueeze is needed for LSTM:
            if model_type == 'QuantileLSTM' or model_type == 'LSTM':
                model_input = X_train.unsqueeze(1)
            else:
                model_input = X_train

            preds_train = model(model_input)
            
            # Use the VECTOR y_priori_train, not a scalar
            loss_train = criterion(preds_train, y_train, weight=weight, preds_prior=y_priori_train)
            
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                if model_type == 'QuantileLSTM' or model_type == 'LSTM':
                    test_input = X_test.unsqueeze(1)
                else:
                    test_input = X_test
                
                preds_test = model(test_input)
                loss_test = criterion(preds_test, y_test, weight=weight, preds_prior=y_priori_test)
            
            history["train_loss"].append(loss_train.item())
            history["test_loss"].append(loss_test.item())
            
            loss = loss_train.item()
            
            # Check for NaN (Exploding Gradients)
            if np.isnan(loss):
                print(f"❌ Error: Loss is NaN at epoch {epoch}. Reduce LR or Check Data.")
                break

            # Early Stopping (Optional: Relaxed tolerance)
            if np.abs(1 - loss / (loss_prev + 1e-8)) < 0.000001:
                 # Added small epsilon to prevent division by zero
                if not silent: print(f"Early stopping at epoch {epoch}")
                break
            else:
                loss_prev = loss

        X_test_aux.append(X_test)
        y_test_aux.append(y_test)
        preds_test_aux.append(preds_test)
        dates_test_aux.append(dates_test)
        
        split_idx += window_size
        
        # Capture state for next iteration (Warm Start)
        last_model_state = model.state_dict()
        pretrained_state_dict = last_model_state
        
    return model, history, (X_train, y_train), (X_test_aux, y_test_aux, preds_test_aux, dates_test_aux)

def analyze_var_functional(df, rolling=132, alpha=0.05, z0=1.645, 
                              std_range=(0, 2), mean_range=(-1, 2), n_points=31, weight=1):
    """
    Analyze the quantile loss functional for different actual mean and std values.
    
    The theoretical optimal VaR is: mu - z0 * sigma
    We test combinations to see which minimizes the loss.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'log_ret' column
    rolling : int
        Rolling window size for calculating mean and std
    alpha : float
        Significance level (e.g., 0.05 for 95% confidence)
    z0 : float
        Z-score for the given alpha (e.g., 1.645 for alpha=0.05)
    std_range : tuple
        Range for std multipliers (min, max). Default: (0, 2)
    mean_range : tuple
        Range for mean multipliers (min, max). Default: (-1, 2)
    n_points : int
        Number of points to test in each dimension. Default: 31
    
    Returns:
    --------
    tuple : results_df, loss_matrix, actual_std_values, actual_mean_values, 
            optimal_params, theoretical_loss, avg_mean, avg_std
    """
    # Prepare data
    df_toy = df.copy()
    df_toy['std'] = df_toy['log_ret'].rolling(rolling).std()
    df_toy['mean'] = df_toy['log_ret'].rolling(rolling).mean()
    df_toy['log_ret_p1'] = df_toy['log_ret'].shift(-1)
    df_toy = df_toy.dropna()
    
    # Calculate average mean and std to create meaningful ranges
    avg_mean = df_toy['mean'].mean()
    avg_std = df_toy['std'].mean()
    var_priori = avg_mean - z0 * avg_std
    
    print(f"Average rolling mean: {avg_mean:.6f}")
    print(f"Average rolling std: {avg_std:.6f}")
    print(f"Testing std multipliers from {std_range[0]} to {std_range[1]}")
    print(f"Testing mean multipliers from {mean_range[0]} to {mean_range[1]}")
    print(f"Grid size: {n_points} × {n_points} = {n_points**2} points")
    
    # Define grid of multipliers using the provided ranges
    std_multipliers = np.linspace(std_range[0], std_range[1], n_points)
    mean_multipliers = np.linspace(mean_range[0], mean_range[1], n_points)
    
    # Convert to actual values for plotting
    actual_std_values = avg_std * std_multipliers
    actual_mean_values = avg_mean * mean_multipliers
    
    # Store results
    results = []
    loss_matrix = np.zeros((len(std_multipliers), len(mean_multipliers)))
    
    # Calculate loss for each combination
    for i, aux_std in enumerate(std_multipliers):
        for j, aux_mean in enumerate(mean_multipliers):
            # Calculate VaR estimate using the multipliers
            var_estimate = -z0 * df_toy['std'] * aux_std + df_toy['mean'] * aux_mean
            
            # Calculate quantile loss
            loss = calculate_quantile_loss(df_toy['log_ret_p1'].values, 
                                          var_estimate.values, 
                                          alpha, weight, var_priori)
            
            loss_matrix[i, j] = loss
            results.append({
                'aux_std': aux_std,
                'aux_mean': aux_mean,
                'actual_std': avg_std * aux_std,
                'actual_mean': avg_mean * aux_mean,
                'loss': loss
            })
    
    results_df = pd.DataFrame(results)
    
    # Find optimal values
    optimal_idx = results_df['loss'].idxmin()
    optimal_params = results_df.loc[optimal_idx]
    
    # Find loss at theoretical optimal (aux_std=1, aux_mean=1)
    theoretical_idx = results_df[(results_df['aux_std'] == 1.0) & 
                                  (results_df['aux_mean'] == 1.0)].index[0]
    theoretical_loss = results_df.loc[theoretical_idx, 'loss']
    
    return results_df, loss_matrix, actual_std_values, actual_mean_values, optimal_params, theoretical_loss, avg_mean, avg_std



def calculate_quantile_loss(returns, var_estimates, alpha=0.05, weight=1, var_priori=0):
    """
    Calculate the quantile loss (pinball loss) for VaR estimates.
    
    Loss = alpha * (y - var) if y >= var
    Loss = (alpha - 1) * (y - var) if y < var
    """
    errors = returns - var_estimates
    loss = np.where(errors >= 0, 
                    alpha * errors, 
                    (alpha - 1) * errors)
    
    return np.mean(loss + weight * (var_priori - var_estimates)**2)
