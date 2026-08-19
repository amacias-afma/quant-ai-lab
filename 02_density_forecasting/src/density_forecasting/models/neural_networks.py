"""
Neural Network Models and Training Utilities for Density Forecasting.

This module contains:
  - Reproducibility helpers (set_seed)
  - PyTorch network architectures for Student-t density forecasting
    (BasicStudentTNet, ResidualStudentTNet, NeuralSDE_StudentT,
     NeuralSDE_Density)
  - Loss / objective functions (student_t_nll, anchored_student_t_loss, …)
  - Walk-forward backtesting engine (run_model)
  - Monte Carlo ensemble generator (generate_montecarlo)
  - Path-signature feature extractor (extract_signatures)
"""

import copy
import random

import numpy as np
import esig
import scipy.stats as stats

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.studentT import StudentT

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Lock all random engines for complete reproducibility.

    Sets the seed for Python's ``random``, NumPy, and all PyTorch random
    number generators (CPU and every CUDA device), and forces PyTorch to
    use deterministic algorithms where available.

    Args:
        seed: Integer seed value.  Defaults to 42.

    Note:
        Call this function **before** creating your model and data loaders
        to ensure a fully reproducible run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Forces PyTorch to use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Network architectures
# ---------------------------------------------------------------------------

class ResidualStudentTNet(nn.Module):
    """Residual Student-t density network with historical-prior anchoring.

    The network predicts *deltas* on top of historical Student-t parameters
    (mu, sigma, nu) that are passed as the first three features of the input
    tensor.  This residual design biases the model toward the rolling
    statistical baseline while still allowing data-driven corrections.

    Architecture:
        - Shared hidden layers: Linear(input_dim→16) → ReLU → Dropout(0.3)
          → Linear(16→8) → ReLU → Dropout(0.3)
        - Three delta heads: Linear(8→1) each for Δμ, Δσ, Δν
        - Output heads initialised with tiny uniform weights so predictions
          start close to the historical prior.

    Args:
        input_dim: Number of input features.  The first three features MUST
                   be (mu_hist, sigma_hist, nu_hist) in that order.
    """

    def __init__(self, input_dim: int) -> None:
        super(ResidualStudentTNet, self).__init__()

        hidden_neurons = 16
        # Keep the hidden layers tiny to prevent overfitting
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_neurons),
            nn.ReLU(),
            nn.Dropout(0.3),  # Essential for financial data
            nn.Linear(hidden_neurons, 8),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # These heads predict the DELTA (the change), not the absolute value
        self.delta_mu = nn.Linear(8, 1)
        self.delta_sigma = nn.Linear(8, 1)
        self.delta_nu = nn.Linear(8, 1)

        # Initialise the delta layers to output numbers very close to 0 at the start
        nn.init.uniform_(self.delta_mu.weight, -0.01, 0.01)
        nn.init.uniform_(self.delta_sigma.weight, -0.01, 0.01)
        nn.init.uniform_(self.delta_nu.weight, -0.01, 0.01)

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, input_dim).  The first three
               columns must be (mu_hist, sigma_hist, nu_hist).

        Returns:
            Tuple (mu_pred, sigma_pred, nu_pred), each of shape (batch, 1).
            sigma_pred and nu_pred are guaranteed positive via Softplus.
        """
        import torch.nn.functional as F

        # x contains: [mu_hist, sigma_hist, nu_hist, VIX, lag1, lag2, lag3, lag4, lag5]
        mu_hist    = x[:, 0:1]
        sigma_hist = x[:, 1:2]
        nu_hist    = x[:, 2:3]

        # Pass the full feature vector through the hidden layers
        h = self.hidden(x)

        # Calculate deltas (can be negative or positive)
        d_mu    = self.delta_mu(h)
        d_sigma = self.delta_sigma(h)
        d_nu    = self.delta_nu(h)

        # RESIDUAL CONNECTION: Output = Historical Prior + Neural Network Delta
        mu_pred    = mu_hist + d_mu
        sigma_pred = F.softplus(sigma_hist + d_sigma) + 1e-6
        nu_pred    = F.softplus(nu_hist + d_nu) + 2.1

        return mu_pred, sigma_pred, nu_pred

class BasicStudentTNet(nn.Module):
    """Simple feed-forward Student-t density network.

    A lightweight baseline architecture with a single shared hidden layer
    and three independent output heads for the Student-t parameters
    (mu, sigma, nu).

    Architecture:
        - Shared hidden layer: Linear(input_dim→hidden_neurons) → ReLU
        - mu head:    Linear(hidden_neurons→1)
        - sigma head: Linear(hidden_neurons→1) → Softplus + ε  (> 0)
        - nu head:    Linear(hidden_neurons→1) → Softplus + 2.1 (> 2)

    Args:
        input_dim:       Number of input features.
        hidden_neurons:  Width of the shared hidden layer.  Defaults to 16.
    """

    def __init__(self, input_dim: int, hidden_neurons: int = 16) -> None:
        super(BasicStudentTNet, self).__init__()

        # Simple shared hidden layer
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_neurons),
            nn.ReLU(),
        )

        # 3 Independent Output Heads
        self.mu_head = nn.Linear(hidden_neurons, 1)

        self.sigma_head = nn.Sequential(
            nn.Linear(hidden_neurons, 1),
            nn.Softplus(),  # Forces sigma to be > 0
        )

        self.nu_head = nn.Sequential(
            nn.Linear(hidden_neurons, 1),
            nn.Softplus(),  # Forces nu to be > 0 (we add 2.1 in the forward pass)
        )

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, input_dim).

        Returns:
            Tuple (mu, sigma, nu), each of shape (batch, 1).
            sigma > 0 and nu > 2 are guaranteed by construction.
        """
        h = self.hidden(x)

        mu    = self.mu_head(h)
        sigma = self.sigma_head(h) + 1e-6  # Epsilon for numerical stability
        nu    = self.nu_head(h) + 2.1      # nu > 2 is required for finite variance

        return mu, sigma, nu


# ---------------------------------------------------------------------------
# Loss / objective functions
# ---------------------------------------------------------------------------

def student_t_nll(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    nu: torch.Tensor,
    actual_return: torch.Tensor,
) -> torch.Tensor:
    """Negative Log-Likelihood of the Student-t distribution.

    Measures how surprised the predicted distribution is by the actual
    observed return.  Minimising this loss trains the network to assign
    high probability to realised returns.

    Args:
        mu:            Predicted location (drift).  Shape: (batch, 1).
        sigma:         Predicted scale (volatility).  Shape: (batch, 1).  Must be > 0.
        nu:            Predicted degrees of freedom.  Shape: (batch, 1).  Must be > 2.
        actual_return: Ground-truth return observations.  Shape: (batch, 1).

    Returns:
        Scalar tensor — the mean NLL over the batch.
    """
    dist = StudentT(df=nu, loc=mu, scale=sigma)
    log_prob = dist.log_prob(actual_return)
    return -log_prob.mean()


# ---------------------------------------------------------------------------
# Walk-forward backtest engine
# ---------------------------------------------------------------------------

def run_model(
    X_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    valid_dates,
    initial_train_size: int,
    step_size: int,
    model_class: str,
) -> tuple:
    """Strict point-in-time walk-forward backtest for Student-t networks.

    Iterates over the dataset in ``step_size`` increments starting from
    ``initial_train_size``.  At each step:

    1. The model is trained on all data strictly before ``current_t``.
    2. The best weights (lowest training loss) are retained via an
       in-loop checkpoint.
    3. Weights from the previous window are used as a warm-start for the
       current window, reducing epochs and learning rate.
    4. Out-of-sample predictions are collected for the next ``step_size``
       time steps.

    Args:
        X_tensor:           Feature tensor, shape (T, n_features).
        y_tensor:           Target tensor, shape (T, 1).
        valid_dates:        DatetimeIndex of length T aligned with the tensors.
        initial_train_size: Number of observations used as the first training
                            set (warm-up period).
        step_size:          Number of new observations added at each
                            walk-forward step (re-training frequency).
        model_class:        Architecture to use.  One of:
                            - ``'Residual'`` → :class:`ResidualStudentTNet`
                            - ``'Basic'``    → :class:`BasicStudentTNet`

    Returns:
        Tuple of five lists/arrays:
        ``(out_of_sample_mu, out_of_sample_sigma, out_of_sample_nu,
        oos_dates, actual_returns)``
    """
    out_of_sample_mu    = []
    out_of_sample_sigma = []
    out_of_sample_nu    = []
    actual_returns      = []
    oos_dates           = valid_dates[initial_train_size:]

    previous_weights = None
    input_dim = X_tensor.shape[1]

    print("Starting Strict Point-in-Time Walk-Forward Backtest...")

    for current_t in range(initial_train_size, len(X_tensor), step_size):
        # 1. Strict historical isolation — no look-ahead
        X_train = X_tensor[:current_t]
        y_train = y_tensor[:current_t]

        # 2. Test window
        end_test = min(current_t + step_size, len(X_tensor))
        X_test   = X_tensor[current_t:end_test]
        y_test   = y_tensor[current_t:end_test]

        # 3. Build model (warm-start when weights are available)
        if model_class == 'Residual':
            print(model_class)
            model = ResidualStudentTNet(input_dim=input_dim)
        else:
            print(model_class)
            model = BasicStudentTNet(input_dim=input_dim, hidden_neurons=8)

        if previous_weights is not None:
            model.load_state_dict(previous_weights)
            # Smaller LR / fewer epochs — already trained
            lr               = 0.001
            epochs_per_step  = 50
        else:
            # Cold start — needs higher LR and more epochs
            lr               = 0.005
            epochs_per_step  = 150

        optimizer      = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        best_loss      = float('inf')
        best_local_weights = None

        # 4. Training loop with in-loop checkpointing
        for epoch in range(epochs_per_step):
            model.train()
            optimizer.zero_grad()

            mu_pred, sigma_pred, nu_pred = model(X_train)
            loss = student_t_nll(mu_pred, sigma_pred, nu_pred, y_train)

            if loss.item() < best_loss:
                best_loss          = loss.item()
                best_local_weights = copy.deepcopy(model.state_dict())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if epoch % 25 == 0:
                print(f"Epoch {epoch:03d} | Current Loss: {loss.item():.4f} | Best Loss: {best_loss:.4f}")

        # 5. Restore best weights
        previous_weights = copy.deepcopy(best_local_weights)
        print(f"\nTraining complete. Restoring best model weights from loss: {best_loss:.4f}")
        model.load_state_dict(best_local_weights)
        model.eval()

        # 6. Out-of-sample predictions
        with torch.no_grad():
            mu_pred, sigma_pred, nu_pred = model(X_test)
            student_t_nll(mu_pred, sigma_pred, nu_pred, y_test)   # kept for logging symmetry

            out_of_sample_mu.extend(mu_pred.numpy().flatten())
            out_of_sample_sigma.extend(sigma_pred.numpy().flatten())
            out_of_sample_nu.extend(nu_pred.numpy().flatten())
            actual_returns.extend(y_test.numpy().flatten())

        print(f"Walk-forward complete up to day {end_test} / {len(X_tensor)}")

    print("Backtest Complete!")
    return out_of_sample_mu, out_of_sample_sigma, out_of_sample_nu, oos_dates, actual_returns


# ---------------------------------------------------------------------------
# Monte Carlo ensemble generator
# ---------------------------------------------------------------------------

def generate_montecarlo(
    out_of_sample_mu: list,
    out_of_sample_sigma: list,
    out_of_sample_nu: list,
    n_samples: int = 1000,
) -> np.ndarray:
    """Draw Monte Carlo ensembles from the predicted Student-t distributions.

    For each out-of-sample time step, draws ``n_samples`` i.i.d. samples
    from the Student-t distribution parameterised by the predicted
    (mu, sigma, nu) triple.  The resulting ensemble matrix is what the
    evaluation metrics (CRPS, PIT, log-likelihood) consume.

    Args:
        out_of_sample_mu:    List of predicted location parameters. Length T_oos.
        out_of_sample_sigma: List of predicted scale parameters.    Length T_oos.
        out_of_sample_nu:    List of predicted degrees of freedom.  Length T_oos.
        n_samples:           Number of Monte Carlo samples per time step.
                             Defaults to 1 000.

    Returns:
        np.ndarray of shape (T_oos, n_samples) where each row i contains
        ``n_samples`` draws from Student-t(nu_i, mu_i, sigma_i).
    """
    mu_arr    = np.array(out_of_sample_mu)
    sigma_arr = np.array(out_of_sample_sigma)
    nu_arr    = np.array(out_of_sample_nu)

    predicted_ensembles = np.zeros((len(mu_arr), n_samples))

    for i in range(len(mu_arr)):
        # scipy.stats.t.rvs takes (df, loc, scale, size)
        predicted_ensembles[i, :] = stats.t.rvs(
            df=nu_arr[i],
            loc=mu_arr[i],
            scale=sigma_arr[i],
            size=n_samples,
        )

    return predicted_ensembles


# ---------------------------------------------------------------------------
# Path-signature feature extractor (legacy / experimental)
# ---------------------------------------------------------------------------

def extract_signatures(df_returns, window_size, signature_depth):
    # X_sigs = []
    # y_targets = []
    # valid_dates = []

    X_combined = []
    y_targets = []
    valid_dates = []

    # Time augmentation: local time for the 30-day window
    local_time = np.linspace(0, 1, window_size)

    returns_array = df_returns['returns'].values
    vix_array = df_returns['VIX'].values
    dates_array = df_returns.index

    print("Extracting Path Signatures...")
    for i in range(len(returns_array) - window_size - 1):
        # 1. The Input: 30 days of returns
        window_returns = returns_array[i : i + window_size]
        window_2d = np.column_stack((local_time, window_returns))
        window_clean = np.ascontiguousarray(window_2d, dtype=np.float64)
        
        # Extract 2D Signature (outputs 7 features)
        sig = esig.stream2sig(window_clean, signature_depth)

        # 2. VIX State (Forward-looking)
        # We only care about the VIX on the last day of the window
        current_vix = vix_array[i + window_size - 1] 
        
        # 3. Combine them! (7 sig features + 1 VIX feature = 8 features)
        combined_features = np.append(sig, current_vix)
        X_combined.append(combined_features)
        
        # window_vix = vix_array[i : i + window_size]
    
        # # 3D Path Augmentation: [Time, BTC Return, VIX]
        # window_3d = np.column_stack((local_time, window_returns, window_vix))
        # window_clean = np.ascontiguousarray(window_3d, dtype=np.float64)
        
        # # Extract Signature
        # sig = esig.stream2sig(window_clean, signature_depth)
        # X_sigs.append(sig)
        
        # 2. The Target: The exact return of tomorrow (Day 31)
        target_return = returns_array[i + window_size]
        y_targets.append(target_return)
        
        # Track dates for plotting later
        valid_dates.append(dates_array[i + window_size])

    X_tensor = torch.tensor(np.array(X_combined), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y_targets), dtype=torch.float32).unsqueeze(1)

    print(f"Feature Matrix Shape: {X_tensor.shape}")
    print(f"Target Matrix Shape: {y_tensor.shape}")
    return X_tensor, y_tensor, valid_dates

class NeuralSDE_StudentT(nn.Module):
    def __init__(self, input_dim):
        super(NeuralSDE_StudentT, self).__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # THREE output heads now
        self.mu_head = nn.Linear(16, 1)
        
        self.sigma_head = nn.Sequential(
            nn.Linear(16, 1),
            nn.Softplus() # Scale must be strictly positive
        )
        
        self.nu_head = nn.Sequential(
            nn.Linear(16, 1),
            nn.Softplus() 
        )
        
    def forward(self, x):
        shared_features = self.shared_net(x)
        
        mu = self.mu_head(shared_features)
        
        # Add epsilon for numerical stability
        sigma = self.sigma_head(shared_features) + 1e-6 
        
        # Nu must be > 2 for variance to be finite. 
        # We add 2.1 to guarantee a stable financial distribution.
        nu = self.nu_head(shared_features) + 2.1 
        
        return mu, sigma, nu

def anchored_student_t_loss(mu, sigma, nu, target, prior_mu, prior_sigma, prior_nu, lambda_reg=0.05):
    """
    Calculates the Student's t NLL with an L2 anchor penalty to historical priors.
    """
    # 1. Negative Log-Likelihood (The Data Fit)
    dist = StudentT(df=nu, loc=mu, scale=sigma)
    nll = -dist.log_prob(target).mean()
    
    # 2. The Anchor Penalty (The Historical Prior)
    # We calculate the mean squared error between the predictions and the priors
    penalty_mu = ((mu - prior_mu) ** 2).mean()
    penalty_sigma = ((sigma - prior_sigma) ** 2).mean()
    penalty_nu = ((nu - prior_nu) ** 2).mean()
    
    total_penalty = penalty_mu + penalty_sigma + penalty_nu
    
    # 3. Combined Objective Function
    return nll + (lambda_reg * total_penalty)
    
def student_t_nll_loss(mu, sigma, nu, target):
    """
    Calculates the Negative Log-Likelihood of a Student's t-distribution.
    Leverages PyTorch's highly optimized internal math.
    """
    # Create the distribution object
    dist = StudentT(df=nu, loc=mu, scale=sigma)
    
    # Calculate the log probability of the actual target happening
    log_prob = dist.log_prob(target)
    
    # We want to MINIMIZE the negative log-likelihood
    return -log_prob.mean()

class NeuralSDE_Density(nn.Module):
    def __init__(self, input_dim):
        super(NeuralSDE_Density, self).__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        # Two output heads: one for Drift (mu), one for Diffusion (sigma)
        self.mu_head = nn.Linear(16, 1)
        self.sigma_head = nn.Sequential(
            nn.Linear(16, 1),
            nn.Softplus() # Sigma must be strictly positive
        )
        
    def forward(self, x):
        shared_features = self.shared_net(x)
        mu = self.mu_head(shared_features)
        
        # Add a tiny epsilon to prevent log(0) errors in the loss function
        sigma = self.sigma_head(shared_features) + 1e-6 
        return mu, sigma

def gaussian_nll_loss(mu, sigma, target):
    """Calculates the Negative Log-Likelihood of a Gaussian distribution."""
    variance = sigma ** 2
    loss = 0.5 * torch.log(variance) + 0.5 * ((target - mu) ** 2) / variance
    return loss.mean()
