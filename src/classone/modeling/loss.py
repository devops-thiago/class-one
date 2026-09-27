"""Proper scoring rules and calibration loss functions for RLCD.

Implements:
- Brier Score Loss (normalized mean squared error of probability distributions)
  Reference: Brier, G. W. (1950). Verification of forecasts expressed in terms of probability.
  Reference: Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation.
- Logarithmic Scoring Rule (Cross-Entropy / Negative Log-Likelihood)
- Expected Calibration Error (ECE)
  Reference: Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks.
- Hybrid RLCD Loss combining log-loss and Brier penalty
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def brier_score_binary(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Computes binary Brier score: (p - y)^2 in [0, 1].

    Args:
        probs: Predicted probability tensor of shape (N,) or (N, 1), in [0, 1].
        targets: Binary label tensor of shape (N,) or (N, 1), in {0, 1}.
    """
    return torch.mean((probs.view(-1).float() - targets.view(-1).float()) ** 2)


def brier_score_multiclass(probs: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Computes normalized multiclass Brier score: 0.5 * sum_k (p_k - y_k)^2 in [0, 1].

    Dividing by 2 is the standard normalization in modern probabilistic forecasting
    and machine learning (e.g. Gneiting & Raftery 2007), ensuring that for K=2 classes
    it reduces identically to binary Brier score (p - y)^2.

    Args:
        probs: Probability distribution tensor of shape (N, K).
        targets: Ground truth class index tensor of shape (N,).
        num_classes: Number of candidate classes K.
    """
    probs_f = probs.float()
    one_hot = F.one_hot(targets, num_classes=num_classes).float()
    return 0.5 * torch.mean(torch.sum((probs_f - one_hot) ** 2, dim=-1))


def expected_calibration_error(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 10) -> float:
    """Calculates the Expected Calibration Error (ECE) across confidence bins.

    Reference: Guo et al. (ICML 2017) "On Calibration of Modern Neural Networks"

    Args:
        probs: Model confidence/probability for the predicted class, shape (N,).
        targets: 1 if prediction is correct, 0 otherwise, shape (N,).
        n_bins: Number of confidence bins.
    """
    probs = probs.view(-1).detach().cpu().float()
    targets = targets.view(-1).detach().cpu().float()

    n_samples = len(probs)
    if n_samples == 0:
        return 0.0

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # For the first bin, include 0.0 with >= to prevent exclusion of zero predictions
        if i == 0:
            in_bin = (probs >= bin_lower) & (probs <= bin_upper)
        else:
            in_bin = (probs > bin_lower) & (probs <= bin_upper)

        count_in_bin = in_bin.sum().item()
        if count_in_bin > 0:
            accuracy_in_bin = targets[in_bin].mean().item()
            avg_confidence_in_bin = probs[in_bin].mean().item()
            ece += abs(avg_confidence_in_bin - accuracy_in_bin) * (count_in_bin / n_samples)

    return float(ece)


class RLCDLoss(nn.Module):
    """Reinforcement Learning for Calibrated Decisions (RLCD) multi-objective loss.

    Combines Logarithmic Scoring (proper scoring rule rewarding true likelihood)
    with Brier Score (penalizing probability deviation).
    """

    def __init__(
        self,
        brier_weight: float = 0.5,
        log_weight: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.brier_weight = brier_weight
        self.log_weight = log_weight
        self.eps = eps

    def forward_binary(self, prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Loss for Noul boolean decisions."""
        prob_f = prob.view(-1).float()
        target_f = target.view(-1).float()
        prob_clamped = torch.clamp(prob_f, self.eps, 1.0 - self.eps)

        # Binary log loss
        log_loss = -(target_f * torch.log(prob_clamped) + (1.0 - target_f) * torch.log(1.0 - prob_clamped)).mean()

        # Un-clamped Brier loss for pure proper scoring
        brier = torch.mean((prob_f - target_f) ** 2)

        return self.log_weight * log_loss + self.brier_weight * brier

    def forward_multiclass(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Loss for Choice / Score decisions."""
        num_classes = probs.shape[-1]
        probs_f = probs.float()
        one_hot = F.one_hot(targets, num_classes=num_classes).float()

        # Log loss computed with stable FP32 clamping
        probs_clamped = torch.clamp(probs_f, self.eps, 1.0 - self.eps)
        log_loss = -torch.sum(one_hot * torch.log(probs_clamped), dim=-1).mean()

        # Normalized multiclass Brier score (0.5 * sum squared difference)
        brier = 0.5 * torch.mean(torch.sum((probs_f - one_hot) ** 2, dim=-1))

        return self.log_weight * log_loss + self.brier_weight * brier
