"""Anchored quantile (pinball) loss.

    L = mean[ max((alpha - 1) * e, alpha * e) ]  +  weight * mean[ (pred - prior)^2 ]
    with e = target - pred

The first term is the standard pinball loss, strictly consistent for the ``alpha``-quantile.
The second is the **anchor**: an L2 pull toward a classical VaR prior (parametric-Normal or
Historical). ``weight = 0`` recovers plain quantile regression — that is the unanchored
ablation, and the whole study is the comparison between the two.

Note the anchor is a *regulariser*, not a physical constraint: it biases the estimator toward
the prior, it does not enforce monotonicity in alpha or sub-additivity.
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["AnchoredQuantileLoss", "QuantileLoss"]


class AnchoredQuantileLoss(nn.Module):
    def __init__(self, alpha: float = 0.05):
        super().__init__()
        self.alpha = alpha

    def forward(self, preds, target, weight=0.0, preds_prior=0.0):
        errors = target - preds
        pinball = torch.max((self.alpha - 1) * errors, self.alpha * errors)
        loss = pinball.mean()
        if weight:
            anchor = (preds - preds_prior) ** 2
            loss = loss + weight * anchor.mean()
        return loss


# Backwards-compatible alias for existing imports.
QuantileLoss = AnchoredQuantileLoss
