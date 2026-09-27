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
