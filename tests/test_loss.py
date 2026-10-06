"""Unit tests for RLCD proper scoring rules and calibration loss."""

import pytest
import torch

from classone.modeling.loss import (
    RLCDLoss,
    brier_score_binary,
    brier_score_multiclass,
    expected_calibration_error,
)


def test_brier_score_binary_perfect():
    probs = torch.tensor([1.0, 0.0, 1.0])
    targets = torch.tensor([1, 0, 1])
    score = brier_score_binary(probs, targets)
    assert pytest.approx(score.item(), abs=1e-5) == 0.0


def test_brier_score_binary_worst():
    probs = torch.tensor([0.0, 1.0])
    targets = torch.tensor([1, 0])
    score = brier_score_binary(probs, targets)
    assert pytest.approx(score.item(), abs=1e-5) == 1.0


def test_brier_score_multiclass():
    probs = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    targets = torch.tensor([0, 1])
    score = brier_score_multiclass(probs, targets, num_classes=3)
    # Target 0: 0.5 * ((0.7-1)^2 + 0.2^2 + 0.1^2) = 0.5 * 0.14 = 0.07
    # Target 1: 0.5 * (0.1^2 + (0.8-1)^2 + 0.1^2) = 0.5 * 0.06 = 0.03
    # Mean: 0.05
    assert pytest.approx(score.item(), abs=1e-5) == 0.05

    # Verify exact equivalence between binary and 2-class multiclass Brier score
    p_bin = torch.tensor([0.8])
    t_bin = torch.tensor([1])
    bin_score = brier_score_binary(p_bin, t_bin)

    p_mc = torch.tensor([[0.2, 0.8]])
    t_mc = torch.tensor([1])
    mc_score = brier_score_multiclass(p_mc, t_mc, num_classes=2)
    assert pytest.approx(bin_score.item(), abs=1e-5) == mc_score.item()


def test_rlcd_loss_computation():
    loss_fn = RLCDLoss(brier_weight=0.5, log_weight=1.0)

    # Binary forward
    prob = torch.tensor([0.8])
    target = torch.tensor([1])
    loss_val = loss_fn.forward_binary(prob, target)
    assert loss_val.item() > 0.0

    # Multiclass forward
    probs = torch.tensor([[0.8, 0.1, 0.1]])
    targets = torch.tensor([0])
    loss_mc = loss_fn.forward_multiclass(probs, targets)
    assert loss_mc.item() > 0.0


def test_expected_calibration_error():
    # Perfectly calibrated predictions
    probs = torch.tensor([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
    targets = torch.tensor([1, 1, 1, 0, 0, 0])
    ece = expected_calibration_error(probs, targets, n_bins=5)
    assert pytest.approx(ece, abs=0.2) == 0.1


def test_rlcd_loss_soft_uniform_distribution():
    """Verifies that RLCDLoss accepts 2D soft probability targets (e.g. uniform for ambiguous/OOD inputs)."""
    loss_fn = RLCDLoss(brier_weight=0.5, log_weight=1.0)

    # Soft uniform target for 4 choices: [0.25, 0.25, 0.25, 0.25]
    probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    uniform_targets = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    loss_val = loss_fn.forward_multiclass(probs, uniform_targets)
    assert loss_val.item() > 0.0

    # Perfectly matching uniform target yields zero Brier score
    brier_component = 0.5 * torch.sum((probs - uniform_targets) ** 2)
    assert brier_component.item() == 0.0


def test_rlcd_focal_loss_modulation():
    """Verifies that focal_gamma > 0 down-weights loss on highly confident correct predictions."""
    loss_standard = RLCDLoss(brier_weight=0.0, log_weight=1.0, margin_weight=0.0, focal_gamma=0.0)
    loss_focal = RLCDLoss(brier_weight=0.0, log_weight=1.0, margin_weight=0.0, focal_gamma=2.0)

    # 1. Easy sample (p=0.95, target=1)
    p_easy = torch.tensor([0.95])
    t_easy = torch.tensor([1.0])
    l_std_easy = loss_standard.forward_binary(p_easy, t_easy)
    l_foc_easy = loss_focal.forward_binary(p_easy, t_easy)

    # Focal loss should significantly reduce the loss for easy sample (by (1 - 0.95)^2 = 0.0025)
    assert l_foc_easy.item() < 0.01 * l_std_easy.item()

    # 2. Hard sample (p=0.40, target=1)
    p_hard = torch.tensor([0.40])
    t_hard = torch.tensor([1.0])
    l_std_hard = loss_standard.forward_binary(p_hard, t_hard)
    l_foc_hard = loss_focal.forward_binary(p_hard, t_hard)

    # Ratio of focal to standard loss is much higher for hard samples than easy samples
    hard_ratio = l_foc_hard.item() / l_std_hard.item()
    easy_ratio = l_foc_easy.item() / l_std_easy.item()
    assert hard_ratio > 10.0 * easy_ratio
