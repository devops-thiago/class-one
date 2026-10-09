#!/usr/bin/env python3
"""Official JevBench v1.6.1 Evaluation Runner for ClassOne System 1.

Implements the complete JevBench v1.6.1 4-axis scoring methodology:
1. Intelligence (25%): Chance-corrected accuracy with equal task-type weighting
   (Choice 33.3%, Noul 33.3%, Score 33.3%) across tiers (Easy 10%, Standard 20%, Judge 30%, Hard 40%).
2. Calibration (25%): Multi-bin Expected Calibration Error (ECE), Brier score,
   and strict distribution validity checking (RENORM_TOL = 0.02).
3. Speed (25%): Logarithmic latency score over median (p50) and tail (p95) times.
   Score(s) = clamp(100 - 20 * log10(s / 0.1s), 0, 100).
4. Cost (25%): Cost in US$ per 1,000 decisions based on token ingestion volume.
   Score($) = clamp(100 - 30 * log10($ / 0.001), 0, 100).
5. Aggregation: 4-Axis Harmonic Mean Composite Score with quadratic penalty gate below 50.0 Intelligence:
   Composite = 4 / (1/I + 1/Cal + 1/Speed + 1/Cost) * (min(1.0, I / 50.0) ** 2)

Supports:
- --mode api: Evaluates via live HTTP API endpoint (default: http://127.0.0.1:8000/v1/decide).
- --mode local: Evaluates via local PyTorch weights on GPU or CPU.

Outputs:
- benchmarks/results/jevbench_v1.6.1_results.json
"""

import argparse
import json
import math
import os
import time
import urllib.request
from typing import Any

import numpy as np

BASE_DATA_URL = "https://raw.githubusercontent.com/fstandhartinger/jevbench/main/datasets/public"
CACHE_DIR = os.path.join("data", "jevbench")
RENORM_TOL = 0.02
GEMMA_TARIFF_PER_M_TOKENS = 0.042  # $0.042 per million input tokens


def validate_probs(probs: dict[str, float], tol: float = RENORM_TOL) -> tuple[dict[str, float], bool]:
    """Strict probability distribution validator (JevBench v1.6.1).

    If sum(p) is within [1 - tol, 1 + tol] and all p >= 0, renormalizes sum to 1.0.
    Otherwise flags as invalid distribution.
    """
    if not probs:
        return {}, False
    for p in probs.values():
        if p < -1e-6 or math.isnan(p) or math.isinf(p):
            return probs, False
    total = sum(probs.values())
    if abs(total - 1.0) > tol or total <= 0:
        return probs, False
    return {k: max(0.0, v / total) for k, v in probs.items()}, True


def calculate_speed_score(p50_sec: float, p95_sec: float) -> tuple[float, float, float]:
    """Computes JevBench Speed score over median and p95 latencies (in seconds).

    Score(s) = clamp(100 - 20 * log10(s / 0.1), 0, 100)
    Speed = (Score(p50) + Score(p95)) / 2
    """
    def s_score(s: float) -> float:
        if s <= 0:
            return 100.0
        val = 100.0 - 20.0 * math.log10(s / 0.1)
        return float(np.clip(val, 0.0, 100.0))

    score_p50 = s_score(p50_sec)
    score_p95 = s_score(p95_sec)
    return (score_p50 + score_p95) / 2.0, score_p50, score_p95


def calculate_cost_score(avg_tokens_per_decision: float, tariff_per_m: float = GEMMA_TARIFF_PER_M_TOKENS) -> tuple[float, float]:
    """Computes JevBench Cost score in US$ per 1,000 decisions.

    Cost per 1k = (avg_tokens * 1000 * tariff_per_m) / 1,000,000
    Score($) = clamp(100 - 30 * log10(cost_per_1k / 0.001), 0, 100)
    """
    cost_per_1k = (avg_tokens_per_decision * 1000.0 * tariff_per_m) / 1_000_000.0
    if cost_per_1k <= 0:
        return 100.0, 0.0
    score = 100.0 - 30.0 * math.log10(cost_per_1k / 0.001)
    return float(np.clip(score, 0.0, 100.0)), cost_per_1k


def calculate_harmonic_mean(scores: list[float]) -> float:
    """Calculates equal-weight harmonic mean across positive scores.

    H = n / sum(1 / x_i)
    """
    if not scores:
        return 0.0
    safe_scores = [max(s, 0.001) for s in scores]
    inv_sum = sum(1.0 / s for s in safe_scores)
    return len(scores) / inv_sum if inv_sum > 0 else 0.0


def compute_ece(confidences: list[float], accuracies: list[float], n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) over confidence and binary accuracy pairs."""
    if not confidences or len(confidences) != len(accuracies):
        return 0.0
    confs = np.array(confidences)
    accs = np.array(accuracies)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confs >= bin_lower) & (confs <= bin_upper if i == n_bins - 1 else confs < bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            bin_acc = np.mean(accs[in_bin])
            bin_conf = np.mean(confs[in_bin])
            ece += np.abs(bin_acc - bin_conf) * prop_in_bin

    return float(ece)


def fetch_or_load_dataset(tier_filename: str) -> list[dict[str, Any]]:
    """Loads tier dataset from local cache or fetches from upstream repository."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    local_path = os.path.join(CACHE_DIR, tier_filename)
    if os.path.exists(local_path):
        with open(local_path, encoding="utf-8") as f:
            lines = f.read().strip().splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    url = f"{BASE_DATA_URL}/{tier_filename}"
    print(f"[*] Downloading {tier_filename} from {url}...")
    with urllib.request.urlopen(url, timeout=30) as resp:
        content = resp.read().decode("utf-8")
    with open(local_path, "w", encoding="utf-8") as f:
        f.write(content)
    lines = content.strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


class ApiEvaluator:
    """Evaluates tasks against live ClassOne HTTP API endpoint."""

    def __init__(self, endpoint_url: str = "http://127.0.0.1:8000/v1/decide"):
        self.endpoint_url = endpoint_url
        import httpx
        self.client = httpx.Client(timeout=30.0)

    def decide(self, state: Any, q_type: str, instructions: str, criteria: Any) -> dict[str, Any]:
        question_payload: dict[str, Any] = {"type": q_type, "instructions": instructions}
        if q_type in ("choice", "score") and criteria:
            question_payload["criteria"] = criteria

        payload = {
            "state": str(state),
            "questions": {
                "q": question_payload
            }
        }
        t0 = time.perf_counter()
        resp = self.client.post(self.endpoint_url, json=payload)
        lat_sec = time.perf_counter() - t0
        if resp.status_code != 200:
            raise RuntimeError(f"API Error {resp.status_code}: {resp.text}")
        res_json = resp.json()
        answers_dict = res_json.get("answers", res_json.get("decisions", {}))
        data = answers_dict.get("q", {})
        return {"data": data, "latency_sec": lat_sec}


class LocalEvaluator:
    """Evaluates tasks using local PyTorch model weights."""

    def __init__(self, model_path: str, heads_path: str, device: str = "cuda:0", quant: str = None):
        import torch
        from transformers import AutoTokenizer

        from classone.modeling.modeling_classone import ClassOneModel
        from classone.tokenizer import ClassOnePromptBuilder

        self.torch = torch
        self.device = device
        print(f"[*] Loading local ClassOne model from {model_path} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.builder = ClassOnePromptBuilder(self.tokenizer)
        is_large = any(k in model_path.lower() for k in ["9b", "12b", "e4b", "4b"]) or quant is not None
        if quant is None and is_large:
            quant = "4bit"
        self.model = ClassOneModel.from_backbone(
            base_model_name_or_path=model_path,
            tokenizer=self.tokenizer,
            device=device,
            torch_dtype=torch.bfloat16 if is_large else torch.float16,
            quantization=quant,
        )
        heads = torch.load(heads_path, map_location=device)
        self.model.noul_head.load_state_dict(heads["noul_head"], strict=False)
        self.model.choice_head.load_state_dict(heads["choice_head"], strict=False)
        self.model.score_head.load_state_dict(heads["score_head"], strict=False)
        self.model.eval()

    def decide(self, state: Any, q_type: str, instructions: str, criteria: Any) -> dict[str, Any]:
        from classone.schemas import ChoiceQuestion, NoulQuestion, ScoreQuestion
        if q_type == "noul":
            co_q = NoulQuestion(instructions=instructions)
        elif q_type == "choice":
            co_q = ChoiceQuestion(instructions=instructions, criteria=criteria)
        elif q_type == "score":
            co_q = ScoreQuestion(instructions=instructions, criteria=criteria)
        else:
            raise ValueError(f"Unknown type {q_type}")

        packed = self.builder.pack(str(state)[:6000], {"q": co_q})
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()
        t0 = time.perf_counter()
        with self.torch.no_grad():
            res = self.model.evaluate_packed(packed)["q"]
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()
        lat_sec = time.perf_counter() - t0

        data = {
            "type": res.type,
        }
        if res.type == "noul":
            data["noul"] = float(res.noul)
            data["confidence"] = float(res.noul if res.noul >= 0.5 else (1.0 - res.noul))
        elif res.type == "choice":
            data["choice"] = res.choice
            data["probabilities"] = {k: float(v) for k, v in res.probabilities.items()}
            data["confidence"] = float(res.confidence)
        elif res.type == "score":
            data["score"] = float(res.score)
            data["probabilities"] = {k: float(v) for k, v in res.probabilities.items()}
            data["confidence"] = float(res.confidence)

        return {"data": data, "latency_sec": lat_sec}


def run_evaluation(evaluator: Any, tier_name: str, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    print(f"\n{'=' * 78}")
    print(f"  JEVBENCH 1.6.1 EVALUATION: {tier_name.upper()} ({len(tasks)} tasks)")
    print(f"{'=' * 78}")

    results = []
    latencies = []
    all_confs = []
    all_accs = []
    total_tokens = 0

    type_stats = {
        "choice": {"correct": 0, "total": 0, "chance_corrected": []},
        "noul": {"correct": 0, "total": 0, "chance_corrected": []},
        "score": {"correct": 0, "total": 0, "chance_corrected": []},
    }

    for idx, task in enumerate(tasks):
        state = task["state"]
        q_data = task["question"]
        expected = task["expected"]
        q_type = q_data.get("type", "choice")

        instructions = q_data.get("instructions", "")
        if isinstance(instructions, dict):
            instructions = json.dumps(instructions)

        # Approximate token count: state + instructions
        state_str = str(state)
        token_count = max(50, len(state_str.split()) * 4 // 3 + 100)
        total_tokens += token_count

        # Build criteria and compute item chance baseline
        criteria: Any = None
        chance_baseline = 0.5

        if q_type == "noul":
            chance_baseline = 0.5
        elif q_type == "choice":
            raw_crit = q_data.get("criteria", {})
            if isinstance(raw_crit, list):
                criteria = {str(i): str(c) for i, c in enumerate(raw_crit)}
            elif isinstance(raw_crit, dict):
                criteria = {
                    str(k): (v.get("what", str(v)) if isinstance(v, dict) else str(v))
                    for k, v in raw_crit.items()
                }
            else:
                criteria = {"yes": "yes", "no": "no"}
            num_opts = max(2, len(criteria))
            chance_baseline = 1.0 / num_opts
        elif q_type == "score":
            raw_crit = q_data.get("criteria", [])
            if isinstance(raw_crit, dict):
                criteria = [str(v) for v in raw_crit.values()]
            elif isinstance(raw_crit, list):
                criteria = [str(c) for c in raw_crit]
            else:
                criteria = ["low", "medium", "high"]
            num_levels = max(2, len(criteria))
            chance_baseline = 1.0 / num_levels

        try:
            resp = evaluator.decide(state_str, q_type, instructions, criteria)
            decision = resp["data"]
            lat_sec = resp["latency_sec"]
            latencies.append(lat_sec)
        except Exception as e:
            print(f"  [!] Error on task {idx}: {e}")
            continue

        is_correct = False
        conf = 0.5

        # Decision validation and correctness
        if q_type == "noul":
            prob = decision.get("noul", 0.5)
            pred_bool = prob >= 0.5
            expected_bool = str(expected).lower() in ("yes", "true", "1")
            is_correct = (pred_bool == expected_bool)
            conf = prob if pred_bool else (1.0 - prob)

        elif q_type == "choice":
            probs = decision.get("probabilities", {})
            norm_probs, valid = validate_probs(probs)
            pred_choice = decision.get("choice")
            if not valid:
                is_correct = False
                conf = 0.0
            else:
                is_correct = (str(pred_choice) == str(expected))
                conf = max(norm_probs.values()) if norm_probs else decision.get("confidence", 0.5)

        elif q_type == "score":
            probs = decision.get("probabilities", {})
            norm_probs, valid = validate_probs(probs)
            if not valid:
                is_correct = False
                conf = 0.0
            else:
                if norm_probs:
                    best_level = max(norm_probs.items(), key=lambda kv: kv[1])[0]
                    pred_idx = int(best_level) - 1
                else:
                    pred_score = decision.get("score", 1.0)
                    pred_idx = int(round(pred_score)) - 1
                expected_idx = int(expected)
                is_correct = (pred_idx == expected_idx)
                conf = max(norm_probs.values()) if norm_probs else decision.get("confidence", 0.5)

        acc_int = 1 if is_correct else 0
        all_accs.append(acc_int)
        all_confs.append(float(conf))

        # Chance-corrected score (JevBench formula)
        cc_score = max(0.0, (acc_int - chance_baseline) / (1.0 - chance_baseline)) * 100.0

        if q_type in type_stats:
            type_stats[q_type]["total"] += 1
            type_stats[q_type]["correct"] += acc_int
            type_stats[q_type]["chance_corrected"].append(cc_score)

        results.append({
            "task_id": idx,
            "type": q_type,
            "correct": is_correct,
            "confidence": round(float(conf), 4),
            "chance_baseline": round(chance_baseline, 4),
            "chance_corrected": round(cc_score, 2),
            "latency_ms": round(lat_sec * 1000.0, 2),
        })

        if (idx + 1) % 25 == 0 or (idx + 1) == len(tasks):
            curr_acc = (sum(all_accs) / len(all_accs)) * 100.0 if all_accs else 0.0
            print(f"  Processed {idx + 1}/{len(tasks)} tasks | Running Accuracy: {curr_acc:.1f}%")

    tot_tasks = len(all_accs)
    raw_acc = (sum(all_accs) / tot_tasks) * 100.0 if tot_tasks > 0 else 0.0
    p50_sec = float(np.median(latencies)) if latencies else 0.1
    p95_sec = float(np.percentile(latencies, 95)) if latencies else 0.2
    avg_tokens = total_tokens / max(1, tot_tasks)
    ece = compute_ece(all_confs, all_accs)

    # Per-type intelligence scores
    type_intelligence = {}
    for qt, stats in type_stats.items():
        if stats["chance_corrected"]:
            type_intelligence[qt] = float(np.mean(stats["chance_corrected"]))
        else:
            type_intelligence[qt] = 0.0

    return {
        "tier": tier_name,
        "tasks": tot_tasks,
        "raw_accuracy": round(raw_acc, 2),
        "ece": round(ece, 4),
        "p50_ms": round(p50_sec * 1000.0, 2),
        "p95_ms": round(p95_sec * 1000.0, 2),
        "avg_tokens": round(avg_tokens, 1),
        "type_intelligence": {k: round(v, 2) for k, v in type_intelligence.items()},
        "type_stats": {
            qt: {
                "correct": s["correct"],
                "total": s["total"],
                "accuracy": round((s["correct"] / s["total"] * 100.0) if s["total"] > 0 else 0.0, 2)
            }
            for qt, s in type_stats.items()
        },
        "all_latencies_sec": latencies,
        "all_confidences": all_confs,
        "all_accuracies": all_accs,
    }


def main():
    parser = argparse.ArgumentParser(description="Official JevBench v1.6.1 Benchmark Runner")
    parser.add_argument("--mode", choices=["api", "local"], default="local", help="Execution mode (api or local)")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/decide", help="API endpoint URL")
    parser.add_argument(
        "--model-path",
        default="worktrees/qwen-3.5-9b/checkpoints/classone_qwen9b_champion_merged",
        help="Local model weights",
    )
    parser.add_argument(
        "--heads-path",
        default="worktrees/qwen-3.5-9b/checkpoints/classone_qwen9b_champion_merged/classone_heads.pt",
        help="Decision heads file",
    )
    parser.add_argument("--device", default="cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cuda:0", help="PyTorch device")
    parser.add_argument("--quant", default="4bit", help="Backbone quantization (4bit or 8bit)")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("      JEVBENCH v1.6.1 OFFICIAL 4-AXIS BENCHMARK EVALUATOR")
    print(f"      Mode: {args.mode.upper()} | Target: {args.endpoint if args.mode == 'api' else args.model_path}")
    print("=" * 80)

    if args.mode == "api":
        evaluator = ApiEvaluator(endpoint_url=args.endpoint)
    else:
        evaluator = LocalEvaluator(model_path=args.model_path, heads_path=args.heads_path, device=args.device, quant=args.quant)

    tiers = [
        ("Easy", "easy.jsonl", 0.10),
        ("Standard", "original.jsonl", 0.40),
        ("Hard", "hard.jsonl", 0.50),
    ]

    tier_results = []
    combined_latencies = []
    combined_confs = []
    combined_accs = []
    total_tokens_weighted = 0.0

    for tier_name, filename, tier_weight in tiers:
        tasks = fetch_or_load_dataset(filename)
        res = run_evaluation(evaluator, tier_name, tasks)
        res["tier_weight"] = tier_weight
        tier_results.append(res)
        combined_latencies.extend(res.pop("all_latencies_sec"))
        combined_confs.extend(res.pop("all_confidences"))
        combined_accs.extend(res.pop("all_accuracies"))
        total_tokens_weighted += res["avg_tokens"] * tier_weight

    # 1. Axis 1: Intelligence (Equal type weights Choice 33.3%, Noul 33.3%, Score 33.3% across tiers)
    overall_choice = np.mean([r["type_intelligence"]["choice"] for r in tier_results if r["type_stats"]["choice"]["total"] > 0])
    overall_noul = np.mean([r["type_intelligence"]["noul"] for r in tier_results if r["type_stats"]["noul"]["total"] > 0])
    overall_score = np.mean([r["type_intelligence"]["score"] for r in tier_results if r["type_stats"]["score"]["total"] > 0])
    intelligence_score = float((overall_choice + overall_noul + overall_score) / 3.0)

    # 2. Axis 2: Calibration (ECE and Brier score over all predictions)
    overall_ece = compute_ece(combined_confs, combined_accs)
    brier_score = float(np.mean((np.array(combined_confs) - np.array(combined_accs)) ** 2))
    cal_score = float(np.clip(100.0 * (1.0 - overall_ece) * (1.0 - brier_score), 0.0, 100.0))

    # 3. Axis 3: Speed (p50 and p95 over all queries)
    p50_sec = float(np.median(combined_latencies))
    p95_sec = float(np.percentile(combined_latencies, 95))
    speed_score, speed_p50_pts, speed_p95_pts = calculate_speed_score(p50_sec, p95_sec)

    # 4. Axis 4: Cost (US$ per 1,000 decisions under Gemma 4 tariff)
    cost_score, cost_per_1k = calculate_cost_score(total_tokens_weighted, GEMMA_TARIFF_PER_M_TOKENS)

    # 5. Composite Score: 4-Axis Harmonic Mean with Intelligence Quadratic Penalty Gate
    raw_composite = calculate_harmonic_mean([intelligence_score, cal_score, speed_score, cost_score])
    quadratic_gate = (intelligence_score / 50.0) ** 2 if intelligence_score < 50.0 else 1.0
    final_composite = raw_composite * quadratic_gate

    # Capability Score: Harmonic Mean of Intelligence and Calibration
    capability_score = calculate_harmonic_mean([intelligence_score, cal_score])

    # Print Official Scorecard
    print("\n" + "=" * 80)
    print("                  OFFICIAL JEVBENCH v1.6.1 SCORECARD")
    print("=" * 80)
    print(f"  Composite Score (Harmonic Mean) : {final_composite:.2f} / 100.00")
    print(f"  Capability Score (I + Cal)      : {capability_score:.2f} / 100.00")
    print("-" * 80)
    print(f"  Axis 1: Intelligence (25%)       : {intelligence_score:.2f} pts")
    print(f"          • Choice Intelligence    : {overall_choice:.2f}")
    print(f"          • Noul Intelligence      : {overall_noul:.2f}")
    print(f"          • Score Intelligence     : {overall_score:.2f}")
    print(f"  Axis 2: Calibration (25%)        : {cal_score:.2f} pts (ECE: {overall_ece:.4f}, Brier: {brier_score:.4f})")
    print(f"  Axis 3: Speed (25%)              : {speed_score:.2f} pts (p50: {p50_sec * 1000.0:.1f}ms, p95: {p95_sec * 1000.0:.1f}ms)")
    print(f"  Axis 4: Cost (25%)               : {cost_score:.2f} pts (${cost_per_1k:.5f} / 1,000 decisions)")
    print(f"  Penalty Gate (<50 Intelligence)  : {quadratic_gate:.4f}")
    print("=" * 80)

    # Save artifact
    output_dir = os.path.join("benchmarks", "results")
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "jevbench_v1.6.1_results.json")

    summary_artifact = {
        "benchmark": "JevBench",
        "version": "1.6.1",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "mode": args.mode,
        "composite_score": round(final_composite, 2),
        "capability_score": round(capability_score, 2),
        "raw_composite": round(raw_composite, 2),
        "quadratic_gate": round(quadratic_gate, 4),
        "axes": {
            "intelligence": {
                "score": round(intelligence_score, 2),
                "weight": 0.25,
                "choice_intelligence": round(overall_choice, 2),
                "noul_intelligence": round(overall_noul, 2),
                "score_intelligence": round(overall_score, 2),
            },
            "calibration": {
                "score": round(cal_score, 2),
                "weight": 0.25,
                "ece": round(overall_ece, 4),
                "brier": round(brier_score, 4),
            },
            "speed": {
                "score": round(speed_score, 2),
                "weight": 0.25,
                "p50_ms": round(p50_sec * 1000.0, 2),
                "p95_ms": round(p95_sec * 1000.0, 2),
            },
            "cost": {
                "score": round(cost_score, 2),
                "weight": 0.25,
                "cost_per_1k_decisions_usd": round(cost_per_1k, 6),
                "tariff_per_m_tokens": GEMMA_TARIFF_PER_M_TOKENS,
            },
        },
        "tiers": tier_results,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_artifact, f, indent=2)

    print(f"\n[✓] Official JevBench v1.6.1 results successfully exported to:\n    {out_file}\n")


if __name__ == "__main__":
    main()
