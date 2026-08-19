"""Quantile-forecasting architectures.

Every architecture takes a feature tensor and returns one VaR value per row, shape
``(batch, 1)``. Whether the input must be reshaped to a sequence is declared once in
``models.registry`` — never re-derived with ``isinstance`` checks in the training loop.

To add a model: define the ``nn.Module`` here, then add one entry to ``models/registry.py``.
Nothing else in the codebase needs to change.
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["SimpleQuantileNeuron", "QuantileMLP", "QuantileLSTM"]


class SimpleQuantileNeuron(nn.Module):
    """A single linear unit: this is **linear quantile regression** fit by SGD.

    Kept deliberately as the *linear ablation* rung of the benchmark ladder. If a deeper
    model does not beat this on identical features, the gains came from feature
    construction, not from nonlinearity.
    """

    def __init__(self, input_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)


class QuantileMLP(nn.Module):
    """Feed-forward network — the genuinely nonlinear rung.

    ``hidden_size`` and ``num_layers`` describe the hidden stack; the head maps to a single
    quantile. Nonlinearity is what distinguishes this from ``SimpleQuantileNeuron``, so the
    MLP-vs-neuron comparison is the paper's nonlinearity ablation.
    """

    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        d_in = input_size
        for _ in range(max(1, num_layers)):
            layers.append(nn.Linear(d_in, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d_in = hidden_size
        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class QuantileLSTM(nn.Module):
    """Recurrent rung. Expects ``(batch, seq, features)``; uses the last step's hidden state."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
