"""DEPRECATED — kept only so older imports keep working.

The contents moved to focused modules:

    architectures.py   SimpleQuantileNeuron, QuantileMLP, QuantileLSTM
    losses.py          AnchoredQuantileLoss (alias: QuantileLoss)
    train.py           set_seed, train_model

The quantile-loss surface helpers that used to live here (``analyze_var_functional``,
``calculate_quantile_loss``) were duplicates of ``utils.var_functional_analysis``; that module
is now the single source. Import from the new locations in new code.
"""
from __future__ import annotations

import warnings

from value_at_risk.models.deep_var.architectures import (  # noqa: F401
    SimpleQuantileNeuron,
    QuantileMLP,
    QuantileLSTM,
)
from value_at_risk.models.deep_var.losses import (  # noqa: F401
    AnchoredQuantileLoss,
    QuantileLoss,
)
from value_at_risk.models.deep_var.train import set_seed, train_model  # noqa: F401
from value_at_risk.utils.var_functional_analysis import (  # noqa: F401
    analyze_var_functional_v2,
    calculate_quantile_loss,
)

warnings.warn(
    "value_at_risk.models.deep_var.lstm_model is deprecated; import from "
    "architectures / losses / train instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "SimpleQuantileNeuron", "QuantileMLP", "QuantileLSTM",
    "AnchoredQuantileLoss", "QuantileLoss",
    "set_seed", "train_model",
    "analyze_var_functional_v2", "calculate_quantile_loss",
]
