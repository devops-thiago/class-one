#!/usr/bin/env python3
"""Hugging Face Decision Index Benchmark Runner (multimodalart/jev-decision-index).

Evaluates typed decision models across all 5 canonical capability domains:
1. Knowledge & Verification (25.8% weight): Fact checking, science accuracy, entity verification.
2. Language & Policy Contracts (25.8% weight): Banking intent triage, NLI, legal policy adherence.
3. Retrieval & Relevance Triage (20.0% weight): Passage relevance, clause triage, deduplication.
4. Tools & Execution Guardrails (18.3% weight): Tool arguments, prompt injection, action gating.
5. Rubric & Quality Scoring (10.0% weight): Churn risk, severity tiers, policy violation rubrics.

Exports official submission artifacts:
- benchmarks/results/decision_index/scores.json (Radar chart metrics and overall Decision Index)
- benchmarks/results/decision_index/index.json (Hugging Face Space submission manifest)

Supports:
- --mode api: Evaluates via live HTTP API endpoint (default: http://127.0.0.1:8000/v1/decide).
- --mode local: Evaluates via local PyTorch weights.
"""

import argparse
import json
import os
import time
from typing import Any

import numpy as np

DOMAINS_CONFIG = {
    "knowledge": {
        "display_name": "Knowledge & Verification",
        "weight": 0.258,
        "filename": "knowledge.jsonl",
    },
    "language": {
        "display_name": "Language & Policy Contracts",
        "weight": 0.258,
        "filename": "language.jsonl",
    },
    "retrieval": {
        "display_name": "Retrieval & Relevance Triage",
        "weight": 0.200,
        "filename": "retrieval.jsonl",
    },
    "tools": {
        "display_name": "Tools & Execution Guardrails",
        "weight": 0.183,
        "filename": "tools.jsonl",
    },
    "rubric": {
        "display_name": "Rubric & Quality Scoring",
        "weight": 0.101,  # Sum = 1.000
        "filename": "rubric.jsonl",
    },
}

DATA_DIR = os.path.join("data", "decision_index")
OUTPUT_DIR = os.path.join("benchmarks", "results", "decision_index")


def compute_ece(confidences: list[float], accuracies: list[float], n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE)."""
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


def evaluate_domain(evaluator: Any, domain_key: str, domain_info: dict[str, Any]) -> dict[str, Any]:
    file_path = os.path.join(DATA_DIR, domain_info["filename"])
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Domain dataset not found: {file_path}")

    with open(file_path, encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    print(f"\n[*] Evaluating Domain: {domain_info['display_name']} ({len(tasks)} tasks)...")
    correct_count = 0
    chance_scores = []
    latencies = []
    confidences = []
    accuracies = []

    for task in tasks:
        state = task["state"]
        q_type = task["type"]
        instructions = task["instructions"]
        criteria = task.get("criteria")
        expected = task["expected"]

        # Determine chance baseline
        if q_type == "noul":
            chance_baseline = 0.5
        elif q_type == "choice":
            chance_baseline = 1.0 / max(2, len(criteria))
        elif q_type == "score":
            chance_baseline = 1.0 / max(2, len(criteria))
        else:
            chance_baseline = 0.5

        resp = evaluator.decide(state, q_type, instructions, criteria)
        data = resp["data"]
        lat_sec = resp["latency_sec"]
        latencies.append(lat_sec)

        is_correct = False
        conf = 0.5

        if q_type == "noul":
            prob = data.get("noul", 0.5)
            pred_bool = prob >= 0.5
            expected_bool = bool(expected)
            is_correct = (pred_bool == expected_bool)
            conf = prob if pred_bool else (1.0 - prob)

        elif q_type == "choice":
            pred = data.get("choice")
            is_correct = (str(pred) == str(expected))
            probs = data.get("probabilities", {})
            conf = max(probs.values()) if probs else data.get("confidence", 0.5)

        elif q_type == "score":
            probs = data.get("probabilities", {})
            if probs:
                best_level = max(probs.items(), key=lambda kv: kv[1])[0]
                pred_idx = int(best_level) - 1
            else:
                pred_score = data.get("score", 1.0)
                pred_idx = int(round(pred_score)) - 1
            expected_idx = int(expected)
            is_correct = (pred_idx == expected_idx)
            conf = max(probs.values()) if probs else data.get("confidence", 0.5)

        acc_int = 1 if is_correct else 0
        if is_correct:
            correct_count += 1

        confidences.append(float(conf))
        accuracies.append(acc_int)

        cc = max(0.0, (acc_int - chance_baseline) / (1.0 - chance_baseline)) * 100.0
        chance_scores.append(cc)

    tot = len(tasks)
    raw_acc = (correct_count / tot) * 100.0 if tot > 0 else 0.0
    domain_score = float(np.mean(chance_scores)) if chance_scores else 0.0
    p50_lat = float(np.median(latencies)) * 1000.0 if latencies else 0.0
    ece = compute_ece(confidences, accuracies)

    print(f"    Raw Accuracy: {correct_count}/{tot} ({raw_acc:.1f}%) | Chance-Corrected Score: {domain_score:.2f}% | Latency p50: {p50_lat:.1f}ms")

    return {
        "domain": domain_key,
        "display_name": domain_info["display_name"],
        "weight": domain_info["weight"],
        "total_tasks": tot,
        "correct_tasks": correct_count,
        "raw_accuracy": round(raw_acc, 2),
        "domain_score": round(domain_score, 2),
        "ece": round(ece, 4),
        "latency_p50_ms": round(p50_lat, 2),
        "latencies": latencies,
        "confidences": confidences,
        "accuracies": accuracies,
    }


def main():
    parser = argparse.ArgumentParser(description="Hugging Face Decision Index Runner")
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
    parser.add_argument("--quant", default="4bit", help="Backbone quantization")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("      HUGGING FACE DECISION INDEX BENCHMARK EVALUATOR")
    print("      Space: multimodalart/jev-decision-index")
    print(f"      Mode: {args.mode.upper()} | Target: {args.endpoint if args.mode == 'api' else args.model_path}")
    print("=" * 80)

    if args.mode == "api":
        evaluator = ApiEvaluator(endpoint_url=args.endpoint)
    else:
        evaluator = LocalEvaluator(model_path=args.model_path, heads_path=args.heads_path, device=args.device, quant=args.quant)

    all_domain_results = []
    combined_latencies = []
    combined_confs = []
    combined_accs = []
    weighted_score = 0.0

    for domain_key, info in DOMAINS_CONFIG.items():
        res = evaluate_domain(evaluator, domain_key, info)
        all_domain_results.append(res)
        combined_latencies.extend(res.pop("latencies"))
        combined_confs.extend(res.pop("confidences"))
        combined_accs.extend(res.pop("accuracies"))
        weighted_score += res["domain_score"] * res["weight"]

    overall_ece = compute_ece(combined_confs, combined_accs)
    p50_ms = float(np.median(combined_latencies)) * 1000.0 if combined_latencies else 0.0
    p95_ms = float(np.percentile(combined_latencies, 95)) * 1000.0 if combined_latencies else 0.0

    print("\n" + "=" * 80)
    print("              HUGGING FACE DECISION INDEX OFFICIAL SCORECARD")
    print("=" * 80)
    print(f"  Overall Decision Index: {weighted_score:.2f} / 100.00")
    print(f"  Overall ECE           : {overall_ece:.4f}")
    print(f"  Overall Latency (p50) : {p50_ms:.1f} ms (p95: {p95_ms:.1f} ms)")
    print("-" * 80)
    print("  Domain Breakdown (Radar Chart Metrics):")
    for r in all_domain_results:
        print(f"    • {r['display_name']:<32} (wt: {r['weight']*100:>4.1f}%): {r['domain_score']:>6.2f} pts (Raw Acc: {r['raw_accuracy']}%)")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. scores.json (Radar chart metrics for leaderboard UI)
    scores_artifact = {
        "benchmark": "Hugging Face Decision Index",
        "space": "multimodalart/jev-decision-index",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "model_id": "devops-thiago/classone-qwen3.5-9b-champion",
        "overall_decision_index": round(weighted_score, 2),
        "overall_ece": round(overall_ece, 4),
        "latency_p50_ms": round(p50_ms, 2),
        "latency_p95_ms": round(p95_ms, 2),
        "radar_metrics": {
            r["domain"]: {
                "name": r["display_name"],
                "score": r["domain_score"],
                "weight": r["weight"],
                "raw_accuracy": r["raw_accuracy"],
                "tasks": r["total_tasks"],
            }
            for r in all_domain_results
        },
    }
    scores_path = os.path.join(OUTPUT_DIR, "scores.json")
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(scores_artifact, f, indent=2)

    # 2. index.json (Submission manifest metadata)
    index_artifact = {
        "model_name": "ClassOne System 1 Decision Engine (Qwen 3.5 9B Champion)",
        "model_id": "devops-thiago/classone-qwen3.5-9b-champion",
        "architecture": "Qwen 3.5 9B Backbone + Bilinear Decision Heads",
        "parameters": "9.2B",
        "quantization": "4bit NF4 / BF16",
        "license": "apache-2.0",
        "submission_date": time.strftime("%Y-%m-%d", time.gmtime()),
        "hardware_environment": "2x NVIDIA RTX 5060 Ti 16GB",
        "overall_index": round(weighted_score, 2),
        "domain_scores": {
            r["domain"]: round(r["domain_score"], 2) for r in all_domain_results
        },
        "metrics": {
            "ece": round(overall_ece, 4),
            "latency_p50_ms": round(p50_ms, 2),
        },
    }
    index_path = os.path.join(OUTPUT_DIR, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_artifact, f, indent=2)

    print(f"\n[✓] Official Decision Index submission artifacts generated:\n    • {scores_path}\n    • {index_path}\n")


if __name__ == "__main__":
    main()
