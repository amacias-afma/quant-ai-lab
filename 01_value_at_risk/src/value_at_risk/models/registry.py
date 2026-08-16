"""Model registry — the single place that knows what architectures exist.

Adding a new model is one entry here plus the ``nn.Module`` in
``deep_var/architectures.py``. The training loop, the harness and the CLI all read from this
registry, so nothing else needs editing and no ``if model_type == ...`` checks leak into the
pipeline.

Metadata (names, sequence flag, ladder rung) is plain data and imports without torch, so it
stays unit-testable in environments that have no deep-learning stack. Only ``build_model``
imports torch, and it does so lazily.
"""
from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["ModelInfo", "MODEL_REGISTRY", "available_models", "get_model_info", "build_model"]


@dataclass(frozen=True)
class ModelInfo:
    """Everything the pipeline needs to know about an architecture, without importing it."""
    name: str
    cls_name: str                      # class in deep_var.architectures
    expects_sequence: bool             # True -> input reshaped to (batch, 1, features)
    rung: str                          # ladder position: "linear-ablation" | "nonlinear" | ...
    description: str
    default_kwargs: dict = field(default_factory=dict)


MODEL_REGISTRY: dict[str, ModelInfo] = {
    "SimpleQuantileNeuron": ModelInfo(
        name="SimpleQuantileNeuron",
        cls_name="SimpleQuantileNeuron",
        expects_sequence=False,
        rung="linear-ablation",
        description="Single linear unit = linear quantile regression. The ablation rung: "
                    "any deeper model must beat this on identical features.",
    ),
    "QuantileMLP": ModelInfo(
        name="QuantileMLP",
        cls_name="QuantileMLP",
        expects_sequence=False,
        rung="nonlinear",
        description="Feed-forward network. The nonlinearity rung — MLP vs the linear neuron "
                    "is what licenses any claim about neural networks.",
        default_kwargs={"hidden_size": 32, "num_layers": 2},
    ),
    "QuantileLSTM": ModelInfo(
        name="QuantileLSTM",
        cls_name="QuantileLSTM",
        expects_sequence=True,
        rung="recurrent",
        description="LSTM over the feature sequence; uses the final hidden state.",
        default_kwargs={"hidden_size": 64, "num_layers": 2},
    ),
}


def available_models() -> list[str]:
    """Registered architecture names, for CLI help and validation."""
    return list(MODEL_REGISTRY)


def get_model_info(name: str) -> ModelInfo:
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown model {name!r}; available: {', '.join(available_models())}"
        ) from None


def build_model(name: str, input_size: int, **kwargs):
    """Instantiate a registered architecture. Imports torch lazily.

    Registry ``default_kwargs`` are applied first, then caller overrides. Unsupported kwargs
    (e.g. ``hidden_size`` for the linear neuron) are dropped so callers can pass a uniform
    config dict across models.
    """
    import inspect

    from value_at_risk.models.deep_var import architectures

    info = get_model_info(name)
    cls = getattr(architectures, info.cls_name)
    merged = {**info.default_kwargs, **kwargs}
    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
    filtered = {k: v for k, v in merged.items() if k in accepted}
    return cls(input_size=input_size, **filtered)
