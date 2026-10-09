import json
import math
import os

from scripts.eval_decision_index import DOMAINS_CONFIG, compute_ece
from scripts.eval_jevbench_v16 import (
    calculate_cost_score,
    calculate_harmonic_mean,
    calculate_speed_score,
    validate_probs,
)


def test_jevbench_v16_probability_validation():
    # Valid distribution summing to 1.0
    probs = {"A": 0.6, "B": 0.4}
    norm_probs, valid = validate_probs(probs)
    assert valid is True
    assert math.isclose(sum(norm_probs.values()), 1.0, rel_tol=1e-5)

    # Valid distribution within 2% tolerance (e.g. 1.015) gets renormalized
    probs_near = {"A": 0.61, "B": 0.405}  # sum = 1.015
    norm_probs, valid = validate_probs(probs_near)
    assert valid is True
    assert math.isclose(sum(norm_probs.values()), 1.0, rel_tol=1e-5)

    # Invalid distribution beyond tolerance (sum = 1.08 > 1.02)
    probs_invalid = {"A": 0.7, "B": 0.38}
    _, valid = validate_probs(probs_invalid)
    assert valid is False

    # Negative probability is invalid
    probs_negative = {"A": 1.1, "B": -0.1}
    _, valid = validate_probs(probs_negative)
    assert valid is False


def test_jevbench_v16_speed_score():
    # At 100ms (0.1s), score should be 100
    score, p50_score, p95_score = calculate_speed_score(0.1, 0.1)
    assert math.isclose(score, 100.0, abs_tol=1e-3)

    # At 1.0s (10x slower), score drops by 20 points to 80
    score, p50_score, p95_score = calculate_speed_score(1.0, 1.0)
    assert math.isclose(score, 80.0, abs_tol=1e-3)

    # Fast 50ms latency
    score, p50_score, p95_score = calculate_speed_score(0.05, 0.05)
    assert score == 100.0  # clamped to 100


def test_jevbench_v16_cost_score():
    # Cost per 1,000 decisions at $0.001 should be 100
    score, cost_per_1k = calculate_cost_score(avg_tokens_per_decision=23.8095)
    # 23.8095 * 1000 * 0.042 / 1,000,000 = 0.001
    assert math.isclose(score, 100.0, abs_tol=0.1)

    # 10x more expensive ($0.01 per 1,000) drops by 30 points to 70
    score, cost_per_1k = calculate_cost_score(avg_tokens_per_decision=238.095)
    assert math.isclose(score, 70.0, abs_tol=0.1)


def test_jevbench_v16_harmonic_mean_and_gating():
    # Equal 4 scores
    hm = calculate_harmonic_mean([60.0, 60.0, 60.0, 60.0])
    assert math.isclose(hm, 60.0, abs_tol=1e-3)

    # Penalizes low outliers strongly (arithmetic mean is 80; harmonic mean is 50.0)
    hm_skewed = calculate_harmonic_mean([100.0, 100.0, 100.0, 20.0])
    assert math.isclose(hm_skewed, 50.0, abs_tol=1e-3)

    # Quadratic gate below 50.0
    intel_below = 40.0
    gate = (intel_below / 50.0) ** 2
    assert math.isclose(gate, 0.64, abs_tol=1e-4)


def test_hf_decision_index_domain_weights():
    # Check that domain weights sum to 1.0 (with slight float rounding)
    total_weight = sum(info["weight"] for info in DOMAINS_CONFIG.values())
    assert math.isclose(total_weight, 1.0, abs_tol=1e-3)
    assert len(DOMAINS_CONFIG) == 5


def test_decision_index_datasets_exist_and_valid():
    data_dir = os.path.join("data", "decision_index")
    assert os.path.exists(data_dir)

    for domain_key, info in DOMAINS_CONFIG.items():
        filepath = os.path.join(data_dir, info["filename"])
        assert os.path.exists(filepath), f"Missing dataset: {filepath}"
        with open(filepath, encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) >= 10, f"Domain {domain_key} has fewer than 10 tasks"

        for line in lines:
            task = json.loads(line)
            assert "id" in task
            assert "domain" in task and task["domain"] == domain_key
            assert "type" in task and task["type"] in ("choice", "noul", "score")
            assert "state" in task and len(task["state"]) > 0
            assert "instructions" in task and len(task["instructions"]) > 0
            assert "expected" in task


def test_compute_ece():
    # Perfect calibration
    confs = [1.0, 1.0, 0.0, 0.0]
    accs = [1.0, 1.0, 0.0, 0.0]
    ece = compute_ece(confs, accs)
    assert math.isclose(ece, 0.0, abs_tol=1e-5)

    # Poor calibration
    confs_bad = [0.9, 0.9, 0.9, 0.9]
    accs_bad = [0.0, 0.0, 0.0, 0.0]
    ece_bad = compute_ece(confs_bad, accs_bad)
    assert ece_bad > 0.8
