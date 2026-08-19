"""Walk-forward training loop for anchored quantile models.

One loop for every architecture: the registry declares whether an input needs a sequence
dimension, so there are no per-model branches here. The anchor prior is aligned to the data
once, up front, and sliced positionally alongside X and y.

Note on scope: this function performs the *walk-forward refit* over the block after
``split_type['date']``. Choosing the anchor weight and the architecture is the caller's job
and must happen on VALIDATION only — see ``evaluation.harness``.
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from value_at_risk.models.deep_var.losses import AnchoredQuantileLoss
from value_at_risk.models.deep_var.parametric_model import build_anchor_prior
from value_at_risk.models.registry import build_model, get_model_info

__all__ = ["set_seed", "train_model"]


def set_seed(seed: int = 42) -> None:
    """Fix every RNG that can affect a fit, so a (spec, seed) pair is reproducible."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _anchor_series(df, dates, rolling, alpha, anchor_type):
    """Prior series aligned to ``dates``. Delegates to the torch-free implementation so the
    prior logic (including the falsification controls) can be unit-tested without torch."""
    return build_anchor_prior(df, dates, rolling=rolling, alpha=alpha,
                              anchor_type=anchor_type)


def train_model(
    data,
    model_type: str = "SimpleQuantileNeuron",
    alpha: float = 0.05,
    epochs: int = 500,
    lr: float = 0.01,
    rolling: int = 22,
    split_type: dict | None = None,
    regularization_pm: dict | None = None,
    hidden_size: int = 64,
    num_layers: int = 1,
    pretrained_state_dict=None,
    silent: bool = False,
    seed: int = 42,
    refit_window: int = 22,
    min_epochs: int = 100,
    patience: int = 50,
    tol: float = 1e-9,
):
    """Fit and walk forward, refitting every ``refit_window`` steps with a warm start.

    ``split_type`` is ``{'date': 'YYYY-MM-DD'}`` (walk forward after that date) or
    ``{'percentage': 0.8}`` (single split at that fraction).
    ``regularization_pm`` is ``{'weight': w, 'df': df, 'type': 'param'|'hist'}`` or None.

    Returns ``(model, history, (X_train, y_train), (X_test, y_test, preds, dates))`` where the
    last four are lists over walk-forward blocks.
    """
    set_seed(seed)
    info = get_model_info(model_type)
    split_type = split_type or {"percentage": 0.8}

    X, y, dates = data["X"], data["y"], data["dates"]
    num_samples = len(X)

    # ---- anchor prior, aligned once ------------------------------------------------
    weight = 0.0
    anchor_tensor = None
    if regularization_pm is not None:
        weight = float(regularization_pm["weight"])
        anchor = _anchor_series(
            regularization_pm["df"], dates, rolling, alpha,
            regularization_pm.get("type", "param"),
        )
        valid = ~np.isnan(anchor)
        if not valid.all():                      # drop rows the prior cannot cover
            anchor = anchor[valid]
            X, y = X[valid], y[valid]
            dates = dates[valid].reset_index(drop=True) if isinstance(dates, pd.Series) else dates[valid]
            num_samples = len(X)
        anchor_tensor = torch.tensor(anchor, dtype=torch.float32).unsqueeze(1)

    # ---- split ---------------------------------------------------------------------
    if "percentage" in split_type:
        split_idx = int(num_samples * split_type["percentage"])
        window = num_samples - split_idx
    else:
        split_idx = int((pd.Series(pd.to_datetime(np.asarray(dates)))
                         <= pd.to_datetime(split_type["date"])).sum())
        window = refit_window
    window = max(1, window)
    n_blocks = max(0, (num_samples - split_idx) // window)

    X_test_all, y_test_all, preds_all, dates_all = [], [], [], []
    history = {"train_loss": [], "test_loss": [], "epochs_per_block": []}
    model = None
    X_train = y_train = None

    for _ in range(n_blocks):
        if split_idx >= num_samples:
            break
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:split_idx + window], y[split_idx:split_idx + window]
        if len(X_test) == 0:
            break
        prior_train = anchor_tensor[:split_idx] if anchor_tensor is not None else 0.0
        prior_test = anchor_tensor[split_idx:split_idx + window] if anchor_tensor is not None else 0.0
        block_dates = dates[split_idx:split_idx + window]

        model = build_model(model_type, input_size=X_train.shape[1],
                            hidden_size=hidden_size, num_layers=num_layers)
        if pretrained_state_dict is not None:
            model.load_state_dict(pretrained_state_dict)

        criterion = AnchoredQuantileLoss(alpha=alpha)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_in = X_train.unsqueeze(1) if info.expects_sequence else X_train
        test_in = X_test.unsqueeze(1) if info.expects_sequence else X_test

        # Early stopping: patience on the BEST loss, with a floor of min_epochs.
        #
        # The previous rule broke as soon as two consecutive epochs differed by < 1e-6 in
        # relative terms. For a linear model the pinball objective is piecewise linear, so it
        # genuinely plateaus between steps and that rule fired almost immediately. Adding the
        # L2 anchor makes the objective strictly convex and smooth, so the anchored model did
        # NOT trip the rule and trained far longer than the unanchored one. The anchor was
        # therefore buying training epochs, not statistical information — an optimisation
        # artefact masquerading as a modelling result.
        best_loss = np.inf
        since_improved = 0
        epochs_run = 0
        for epoch in range(epochs):
            model.train()
            preds_train = model(train_in)
            loss_train = criterion(preds_train, y_train, weight=weight, preds_prior=prior_train)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                preds_test = model(test_in)
                loss_test = criterion(preds_test, y_test, weight=weight, preds_prior=prior_test)

            history["train_loss"].append(loss_train.item())
            history["test_loss"].append(loss_test.item())

            loss = loss_train.item()
            epochs_run = epoch + 1
            if np.isnan(loss):
                if not silent:
                    print(f"[warn] NaN loss at epoch {epoch}; stopping this block")
                break
            if loss < best_loss - tol:
                best_loss = loss
                since_improved = 0
            else:
                since_improved += 1
            if epoch + 1 >= min_epochs and since_improved >= patience:
                if not silent:
                    print(f"Early stopping at epoch {epoch} (no improvement in {patience})")
                break

        history["epochs_per_block"].append(epochs_run)

        model.eval()
        with torch.no_grad():
            preds_test = model(test_in)

        X_test_all.append(X_test)
        y_test_all.append(y_test)
        preds_all.append(preds_test)
        dates_all.append(block_dates)

        split_idx += window
        pretrained_state_dict = model.state_dict()      # warm start the next block

    return model, history, (X_train, y_train), (X_test_all, y_test_all, preds_all, dates_all)
