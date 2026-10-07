#!/usr/bin/env python3
"""Evaluates ClassOne on all public JevBench tiers:
1. Easy (datasets/public/easy.jsonl - 48 tasks)
2. Original (datasets/public/original.jsonl - 72 tasks)
3. Hard (datasets/public/hard.jsonl - 111 tasks)

Usage:
    python scripts/eval_jevbench_all_tiers.py [model_path] [heads_path]
"""

import json
import os
import sys
import time
import urllib.request

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from classone.modeling.loss import expected_calibration_error
from classone.modeling.modeling_classone import ClassOneModel
from classone.schemas import ChoiceQuestion, NoulQuestion, ScoreQuestion
from classone.tokenizer import ClassOnePromptBuilder

BASE_URL = "https://raw.githubusercontent.com/fstandhartinger/jevbench/main/datasets/public"
REPO_ID = "devops-thiago/classone-gemma4-e2b"


def fetch_tier_tasks(tier_filename: str) -> list[dict]:
    url = f"{BASE_URL}/{tier_filename}"
    print(f"[*] Downloading {tier_filename} from JevBench repository...")
    with urllib.request.urlopen(url, timeout=30) as resp:
        lines = resp.read().decode("utf-8").strip().splitlines()
    tasks = [json.loads(line) for line in lines if line.strip()]
    print(f"    Loaded {len(tasks)} tasks.")
    return tasks


def evaluate_tier(name: str, tasks: list[dict], model: ClassOneModel, builder: ClassOnePromptBuilder):
    print("\n" + "=" * 76)
    print(f"       EVALUATING JEVBENCH TIER: {name.upper()} ({len(tasks)} tasks)")
    print("=" * 76)

    correct = 0
    total = 0
    latencies = []
    all_confidences = []
    all_accuracies = []

    type_stats = {
        "noul": {"correct": 0, "total": 0},
        "choice": {"correct": 0, "total": 0},
        "score": {"correct": 0, "total": 0},
    }

    for idx, task in enumerate(tasks):
        state = task["state"]
        q_data = task["question"]
        expected = task["expected"]
        q_type = q_data.get("type", "choice")

        instructions = q_data.get("instructions", "")
        if isinstance(instructions, dict):
            instructions = json.dumps(instructions)

        # Build appropriate ClassOne question schema
        if q_type == "noul":
            co_q = NoulQuestion(instructions=instructions)
        elif q_type == "choice":
            raw_crit = q_data.get("criteria", {})
            if isinstance(raw_crit, list):
                criteria = {str(i): str(c) for i, c in enumerate(raw_crit)}
            elif isinstance(raw_crit, dict):
                criteria = {
                    str(k): (v.get("what", str(v)) if isinstance(v, dict) else str(v)) for k, v in raw_crit.items()
                }
            else:
                criteria = {"yes": "yes", "no": "no"}
            co_q = ChoiceQuestion(instructions=instructions, criteria=criteria)
        elif q_type == "score":
            raw_crit = q_data.get("criteria", [])
            if isinstance(raw_crit, dict):
                criteria = [str(v) for v in raw_crit.values()]
            elif isinstance(raw_crit, list):
                criteria = [str(c) for c in raw_crit]
            else:
                criteria = ["low", "high"]
            co_q = ScoreQuestion(instructions=instructions, criteria=criteria)
        else:
            continue

        try:
            packed = builder.pack(state, {"q": co_q})
        except Exception:
            # Handle exceptionally long state by truncation if exceeding context limit
            str_state = str(state)[:6000]
            packed = builder.pack(str_state, {"q": co_q})

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            res = model.evaluate_packed(packed)["q"]
        torch.cuda.synchronize()
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat_ms)

        is_correct = False
        confidence = 0.5

        if q_type == "noul":
            pred_bool = res.noul >= 0.5
            expected_bool = str(expected).lower() in ("yes", "true", "1")
            is_correct = pred_bool == expected_bool
            confidence = res.noul if pred_bool else (1.0 - res.noul)

        elif q_type == "choice":
            is_correct = str(res.choice) == str(expected)
            confidence = max(res.probabilities.values()) if res.probabilities else res.confidence

        elif q_type == "score":
            # Map score expected value to discrete level index (0-indexed in JevBench)
            pred_idx = int(round(res.score)) - 1
            expected_idx = int(expected)
            is_correct = pred_idx == expected_idx
            confidence = max(res.probabilities.values()) if res.probabilities else res.confidence

        int_correct = 1 if is_correct else 0
        correct += int_correct
        total += 1

        all_confidences.append(float(confidence))
        all_accuracies.append(float(int_correct))

        type_stats[q_type]["correct"] += int_correct
        type_stats[q_type]["total"] += 1

        if (idx + 1) % 25 == 0 or (idx + 1) == len(tasks):
            print(f"  Processed {idx + 1}/{len(tasks)} tasks (running accuracy: {(correct / total) * 100:.1f}%)...")

    acc_pct = (correct / total) * 100.0 if total > 0 else 0.0
    mean_lat = float(np.mean(latencies))
    p50_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))

    conf_tensor = torch.tensor(all_confidences, dtype=torch.float32)
    acc_tensor = torch.tensor(all_accuracies, dtype=torch.float32)
    ece_raw = expected_calibration_error(conf_tensor, acc_tensor, n_bins=10)
    ece = float(ece_raw.item()) if hasattr(ece_raw, "item") else float(ece_raw)
    brier = float(torch.mean((conf_tensor - acc_tensor) ** 2).item())

    print(f"\n--- {name.upper()} RESULTS SUMMARY ---")
    print(f"  Overall Accuracy    : {correct}/{total} ({acc_pct:.1f}%)")
    print(f"  Expected Calib Error: {ece:.4f}")
    print(f"  Brier Score         : {brier:.4f}")
    print(f"  Median Latency (p50): {p50_lat:.2f} ms (Mean: {mean_lat:.2f} ms | p95: {p95_lat:.2f} ms)")
    print("  Breakdown by Question Type:")
    for qt, stats in type_stats.items():
        if stats["total"] > 0:
            q_acc = (stats["correct"] / stats["total"]) * 100.0
            print(f"    • {qt.upper():<7}: {stats['correct']}/{stats['total']} ({q_acc:.1f}%)")

    return {
        "tier": name,
        "tasks": total,
        "correct": correct,
        "accuracy": round(acc_pct, 2),
        "ece": round(ece, 4),
        "brier": round(brier, 4),
        "latency_p50_ms": round(p50_lat, 2),
        "latency_mean_ms": round(mean_lat, 2),
        "latency_p95_ms": round(p95_lat, 2),
        "type_breakdown": type_stats,
    }


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/classone_gemma4_e2b_merged"
    heads_path = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/classone_gemma4_e2b/classone_heads.pt"

    if not os.path.exists(model_path):
        model_path = REPO_ID
    if not os.path.exists(heads_path):
        heads_path = hf_hub_download(REPO_ID, "classone_heads.pt")

    print(f"[*] Loading {model_path} on {device} (heads from {heads_path})...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    builder = ClassOnePromptBuilder(tokenizer)

    is_large = any(k in model_path.lower() for k in ["e4b", "12b", "9b", "4b"]) or "--4bit" in sys.argv
    quant = "4bit" if is_large else None

    model = ClassOneModel.from_backbone(
        base_model_name_or_path=model_path,
        tokenizer=tokenizer,
        device=device,
        torch_dtype=torch.bfloat16 if is_large else torch.float16,
        quantization=quant,
    )
    heads = torch.load(heads_path, map_location=device)
    model.noul_head.load_state_dict(heads["noul_head"], strict=False)
    model.choice_head.load_state_dict(heads["choice_head"], strict=False)
    model.score_head.load_state_dict(heads["score_head"], strict=False)
    model.eval()

    tier_files = [
        ("Easy", "easy.jsonl"),
        ("Original", "original.jsonl"),
        ("Hard", "hard.jsonl"),
    ]

    all_results = []
    for tier_name, filename in tier_files:
        tasks = fetch_tier_tasks(filename)
        res = evaluate_tier(tier_name, tasks, model, builder)
        all_results.append(res)

    col1 = 16
    col2 = 10
    col3 = 12
    col4 = 10
    col5 = 10
    col6 = 14
    sep = f"{'─' * col1}┼{'─' * (col2 + 2)}┼{'─' * (col3 + 2)}┼{'─' * (col4 + 2)}┼{'─' * (col5 + 2)}┼{'─' * (col6 + 2)}"

    print("\n" + "=" * 80)
    print("             JEVBENCH COMPLETE MULTI-TIER EVALUATION SUMMARY")
    print("=" * 80)
    print(
        f"{'Tier':<{col1}} │ {'Tasks':>{col2}} │ {'Accuracy':>{col3}} │ {'ECE':>{col4}} │ {'Brier':>{col5}} │ {'Latency p50':>{col6}}"
    )
    print(sep)
    for r in all_results:
        acc_str = f"{r['accuracy']:.1f}% ({r['correct']}/{r['tasks']})"
        lat_str = f"{r['latency_p50_ms']:.1f} ms"
        print(
            f"{r['tier']:<{col1}} │ {r['tasks']:>{col2}} │ {acc_str:>{col3}} │ {r['ece']:>{col4}.4f} │ {r['brier']:>{col5}.4f} │ {lat_str:>{col6}}"
        )
    print(sep)

    tot_tasks = sum(r["tasks"] for r in all_results)
    tot_correct = sum(r["correct"] for r in all_results)
    avg_acc = (tot_correct / tot_tasks) * 100.0 if tot_tasks > 0 else 0.0
    print(
        f"{'ALL TIERS TOTAL':<{col1}} │ {tot_tasks:>{col2}} │ {f'{avg_acc:.1f}% ({tot_correct}/{tot_tasks})':>{col3}} │ {'—':>{col4}} │ {'—':>{col5}} │ {'—':>{col6}}"
    )
    print("=" * 80 + "\n")

    out_path = os.path.join("benchmarks", "results", "jevbench_all_tiers_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[✓] Full multi-tier results exported to: {out_path}")


if __name__ == "__main__":
    main()
