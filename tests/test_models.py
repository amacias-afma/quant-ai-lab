import torch
import pytest
from src.models.volatility.lstm import VolatilityLSTM

def test_lstm_initialization():
    """
    Test that the VolatilityLSTM model initializes with default parameters.
    """
    model = VolatilityLSTM(input_size=2, hidden_size=64, num_layers=1)
    assert isinstance(model, torch.nn.Module)
    # Check if the output layer exists
    assert hasattr(model, 'fc')

def test_lstm_forward_pass():
    """
    Test a forward pass with dummy data to ensure tensor shapes are correct.
    """
    # 1. Setup Model
    input_size = 2
    hidden_size = 32
    seq_length = 10
    batch_size = 5
    
    model = VolatilityLSTM(input_size=input_size, hidden_size=hidden_size)
    
    # 2. Create Dummy Input (Batch Size, Sequence Length, Features)
    dummy_input = torch.randn(batch_size, seq_length, input_size)
    
    # 3. Forward Pass
    output = model(dummy_input)
    
    # 4. Assertions
    # Output should be (Batch Size, 1) because we predict 1 variance value
    assert output.shape == (batch_size, 1)
    # Output should not be NaN
    assert not torch.isnan(output).any()