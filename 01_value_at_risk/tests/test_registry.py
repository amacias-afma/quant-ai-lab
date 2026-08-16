"""Registry tests — metadata must be usable without torch installed."""
import importlib.util

import pytest

from value_at_risk.models.registry import (
    MODEL_REGISTRY, available_models, get_model_info, build_model,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None


def test_expected_models_registered():
    names = available_models()
    assert "SimpleQuantileNeuron" in names
    assert "QuantileMLP" in names          # the nonlinear rung must exist
    assert "QuantileLSTM" in names


def test_ladder_rungs_declared():
    # The linear ablation and a nonlinear rung must both be present, or the study cannot
    # say anything about nonlinearity.
    rungs = {i.rung for i in MODEL_REGISTRY.values()}
    assert "linear-ablation" in rungs
    assert "nonlinear" in rungs


def test_sequence_flag_is_declared_not_inferred():
    assert get_model_info("QuantileLSTM").expects_sequence is True
    assert get_model_info("SimpleQuantileNeuron").expects_sequence is False
    assert get_model_info("QuantileMLP").expects_sequence is False


def test_unknown_model_raises_with_help():
    with pytest.raises(KeyError) as exc:
        get_model_info("NoSuchModel")
    assert "available" in str(exc.value)


def test_registry_entries_are_self_consistent():
    for name, info in MODEL_REGISTRY.items():
        assert info.name == name
        assert info.cls_name and info.description
        assert isinstance(info.default_kwargs, dict)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_build_model_shapes_and_kwarg_filtering():
    import torch

    x = torch.randn(16, 3)
    # Linear neuron ignores hidden_size/num_layers rather than raising.
    lin = build_model("SimpleQuantileNeuron", input_size=3, hidden_size=99, num_layers=5)
    assert lin(x).shape == (16, 1)

    mlp = build_model("QuantileMLP", input_size=3, hidden_size=8, num_layers=2)
    assert mlp(x).shape == (16, 1)
    # MLP must be nonlinear: more than one weight matrix in the stack.
    n_linear = sum(1 for m in mlp.modules() if isinstance(m, torch.nn.Linear))
    assert n_linear >= 2

    lstm = build_model("QuantileLSTM", input_size=3, hidden_size=8, num_layers=1)
    assert lstm(x.unsqueeze(1)).shape == (16, 1)
