#!/usr/bin/env python3
"""Validates fresh downloaded Hugging Face models against official benchmark suites.

Performs end-to-end evaluation:
1. Downloads fresh `classone_heads.pt` and tokenizer from Hugging Face Hub.
2. Verifies SHA256 of candidate decision heads.
3. Loads model backbone directly from HF Hub repository identifier.
4. Runs JevBench (231 tasks across Easy, Original, Hard).
5. Runs Hugging Face Decision Index (50 tasks across 5 domains).
6. (Optional) Runs RLCDAlignBench (100 instances across 10 failure modes).
"""

import argparse
import importlib
import json
import os
import sys
import time
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from classone.modeling.modeling_classone import ClassOneModel
from classone.tokenizer import ClassOnePromptBuilder
from classone.schemas import ChoiceQuestion, NoulQuestion, ScoreQuestion


def get_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if not token and os.path.exists(".env"):
        with open(".env", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("HF_TOKEN="):
                    token = line.strip().split("=", 1)[1].strip().strip("'\"")
                    break
    return token


def sha256_file(path: str) -> str:
    import hashlib
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def compute_auroc(y_true: list[int], y_score: list[float]) -> float:
    y_t = np.asarray(y_true)
    y_s = np.asarray(y_score)
    pos = y_s[y_t == 1]
    neg = y_s[y_t == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    u = sum(np.sum(p > neg) + 0.5 * np.sum(p == neg) for p in pos)
    return float(u / (len(pos) * len(neg)))


def evaluate_jevbench(model, builder, pause_tokens: int = 8) -> dict[str, Any]:
    tiers = [
        ("easy", "data/jevbench/easy.jsonl"),
        ("original", "data/jevbench/original.jsonl"),
        ("hard", "data/jevbench/hard.jsonl"),
    ]
    tier_results = {}
    total_correct = 0
    total_tasks = 0

    print("\n--- Evaluating JevBench Public Multi-Tier Benchmark (231 Tasks) ---")
    for tier_name, tier_path in tiers:
        tasks = []
        with open(tier_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    tasks.append(json.loads(line.strip()))

        correct = 0
        t0 = time.perf_counter()
        for task in tasks:
            state = task["state"]
            q_data = task["question"]
            expected = task["expected"]
            q_type = q_data.get("type", "choice")
            instr = q_data.get("instructions", "")
            if isinstance(instr, dict):
                instr = json.dumps(instr)

            if q_type == "noul":
                co_q = NoulQuestion(instructions=instr)
            elif q_type == "choice":
                raw_crit = q_data.get("criteria", {})
                if isinstance(raw_crit, dict):
                    crit = {str(k): (v.get("what", str(v)) if isinstance(v, dict) else str(v)) for k, v in raw_crit.items()}
                elif isinstance(raw_crit, list):
                    crit = {str(i): str(c) for i, c in enumerate(raw_crit)}
                else:
                    crit = {"yes": "yes", "no": "no"}
                co_q = ChoiceQuestion(instructions=instr, criteria=crit)
            elif q_type == "score":
                raw_crit = q_data.get("criteria", [])
                if isinstance(raw_crit, dict):
                    crit = [str(v) for v in raw_crit.values()]
                elif isinstance(raw_crit, list):
                    crit = [str(c) for c in raw_crit]
                else:
                    crit = ["low", "high"]
                co_q = ScoreQuestion(instructions=instr, criteria=crit)
            else:
                continue

            packed = builder.pack(str(state)[:6000], {"q": co_q}, num_pause_tokens=pause_tokens)
            with torch.no_grad():
                res = model.evaluate_packed(packed)["q"]

            if q_type == "noul":
                pred = res.noul >= 0.5
                exp = str(expected).lower() in ("yes", "true", "1")
                is_correct = pred == exp
            elif q_type == "choice":
                is_correct = str(res.choice) == str(expected)
            elif q_type == "score":
                pred_idx = int(round(res.score)) - 1
                is_correct = pred_idx == int(expected)
            else:
                is_correct = False

            if is_correct:
                correct += 1

        dt = time.perf_counter() - t0
        acc = (correct / len(tasks)) * 100.0
        tier_results[tier_name] = {"correct": correct, "total": len(tasks), "accuracy": acc, "latency_s": dt}
        total_correct += correct
        total_tasks += len(tasks)
        print(f"  [{tier_name.upper():8s}] {correct:3d}/{len(tasks):3d} correct ({acc:5.1f}%) in {dt:5.1f}s")

    overall_acc = (total_correct / total_tasks) * 100.0
    print(f"  [OVERALL ] {total_correct:3d}/{total_tasks:3d} correct ({overall_acc:5.1f}%)")
    return {"overall_accuracy": overall_acc, "total_correct": total_correct, "total_tasks": total_tasks, "tiers": tier_results}


def evaluate_decision_index(model, builder, pause_tokens: int = 8) -> dict[str, Any]:
    di_dir = "data/decision_index"
    di_results = {}
    total_correct = 0
    total_tasks = 0

    print("\n--- Evaluating Hugging Face Decision Index (50 Tasks across 5 Domains) ---")
    for filename in sorted(os.listdir(di_dir)):
        if not filename.endswith(".jsonl"):
            continue
        domain = filename.replace(".jsonl", "")
        tasks = []
        with open(os.path.join(di_dir, filename), "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    tasks.append(json.loads(line.strip()))

        correct = 0
        for task in tasks:
            state = task["state"]
            q_type = task["type"]
            instr = task["instructions"]
            expected = task["expected"]
            if q_type == "noul":
                co_q = NoulQuestion(instructions=instr)
            elif q_type == "choice":
                co_q = ChoiceQuestion(instructions=instr, criteria=task.get("criteria", {}))
            elif q_type == "score":
                co_q = ScoreQuestion(instructions=instr, criteria=task.get("criteria", []))
            else:
                continue

            packed = builder.pack(str(state), {"q": co_q}, num_pause_tokens=pause_tokens)
            with torch.no_grad():
                res = model.evaluate_packed(packed)["q"]

            if q_type == "noul":
                pred = res.noul >= 0.5
                exp = str(expected).lower() in ("yes", "true", "1")
                is_correct = pred == exp
            elif q_type == "choice":
                is_correct = str(res.choice) == str(expected)
            elif q_type == "score":
                is_correct = int(round(res.score)) - 1 == int(expected)
            else:
                is_correct = False

            if is_correct:
                correct += 1

        acc = (correct / len(tasks)) * 100.0
        di_results[domain] = {"correct": correct, "total": len(tasks), "accuracy": acc}
        total_correct += correct
        total_tasks += len(tasks)
        print(f"  [{domain:12s}] {correct:2d}/{len(tasks):2d} ({acc:5.1f}%)")

    overall_di = (total_correct / total_tasks) * 100.0
    print(f"  [OVERALL DI ] {total_correct:2d}/{total_tasks:2d} ({overall_di:5.1f}%)")
    return {"overall_decision_index": overall_di, "total_correct": total_correct, "total_tasks": total_tasks, "domains": di_results}


def evaluate_rlcd_alignbench(model, builder, pause_tokens: int = 8) -> dict[str, Any]:
    print("\n--- Evaluating RLCDAlignBench Alignment & Safety Evaluation (100 Instances) ---")
    rlcd_code_path = os.path.abspath("data/RLCDAlignBench/code")
    if rlcd_code_path not in sys.path:
        sys.path.insert(0, rlcd_code_path)

    batteries = [
        "battery_refusal",
        "battery_a_judge",
        "battery_bc_paired",
        "battery_d_rule",
        "battery_e",
        "battery_f_trajectory",
    ]

    all_samples = []
    for b_name in batteries:
        try:
            mod = importlib.import_module(b_name)
            samples = mod.synthetic_samples() if hasattr(mod, "synthetic_samples") else []
            for s in samples:
                axis = s.get("axis") or s.get("meta", {}).get("axis") or b_name.replace("battery_", "")
                all_samples.append((b_name, axis, s))
        except Exception as exc:
            print(f"[!] Warning: Could not load battery {b_name}: {exc}")

    overall_y_true = []
    overall_y_score = []
    axis_data = {}

    for idx, (b_name, axis, s) in enumerate(all_samples):
        if axis not in axis_data:
            axis_data[axis] = {"y_true": [], "y_score": []}

        state = s.get("state") or s.get("context") or s.get("prompt") or ""
        q_type = s.get("type") or "noul"
        instructions = s.get("instructions") or s.get("instruction") or "Does this violate policy?"
        criteria = s.get("criteria")
        ground_truth = s.get("ground_truth") if "ground_truth" in s else s.get("label", 0)

        if q_type == "noul":
            co_q = NoulQuestion(instructions=instructions)
        elif q_type == "choice":
            co_q = ChoiceQuestion(instructions=instructions, criteria=criteria or {"safe": "Safe", "unsafe": "Unsafe"})
        elif q_type == "score":
            co_q = ScoreQuestion(instructions=instructions, criteria=criteria or ["low", "high"])
        else:
            co_q = NoulQuestion(instructions=instructions)

        packed = builder.pack(str(state)[:6000], {"q": co_q}, num_pause_tokens=pause_tokens)
        with torch.no_grad():
            res = model.evaluate_packed(packed)["q"]

        if res.type == "noul":
            score_val = float(res.noul)
        elif res.type == "choice":
            probs = res.probabilities
            score_val = float(probs.get("unsafe", max(probs.values()) if probs else 0.5))
        elif res.type == "score":
            score_val = float(res.score / 5.0)
        else:
            score_val = 0.5

        y_true_binary = 1 if int(ground_truth) > 0 else 0
        overall_y_true.append(y_true_binary)
        overall_y_score.append(score_val)
        axis_data[axis]["y_true"].append(y_true_binary)
        axis_data[axis]["y_score"].append(score_val)

    # Compute balanced accuracy and AUROC
    y_true_arr = np.array(overall_y_true)
    y_score_arr = np.array(overall_y_score)
    preds = (y_score_arr >= 0.5).astype(int)

    # Balanced accuracy: mean of sensitivity and specificity
    pos_mask = (y_true_arr == 1)
    neg_mask = (y_true_arr == 0)
    sens = np.mean(preds[pos_mask] == 1) if np.sum(pos_mask) > 0 else 0.0
    spec = np.mean(preds[neg_mask] == 0) if np.sum(neg_mask) > 0 else 0.0
    balanced_acc = (sens + spec) / 2.0 * 100.0

    auroc = compute_auroc(overall_y_true, overall_y_score)
    print(f"  [RLCDAlignBench AUROC           ] {auroc:.3f}")
    print(f"  [RLCDAlignBench Balanced Accuracy] {balanced_acc:.1f}%")

    # Print top axis highlights
    for axis, data in sorted(axis_data.items()):
        if len(data["y_true"]) > 0:
            ax_auroc = compute_auroc(data["y_true"], data["y_score"])
            ax_preds = (np.array(data["y_score"]) >= 0.5).astype(int)
            ax_acc = np.mean(ax_preds == np.array(data["y_true"])) * 100.0
            print(f"    - {axis:22s}: {ax_acc:5.1f}% acc | AUROC: {ax_auroc:.3f} (N={len(data['y_true'])})")

    return {"overall_auroc": auroc, "balanced_accuracy": balanced_acc, "total_instances": len(all_samples)}


def main():
    parser = argparse.ArgumentParser(description="Validate fresh HF published models")
    parser.add_argument("--repo-id", type=str, required=True, help="HF Repo ID (e.g. devops-thiago/classone-qwen3.5-9b)")
    parser.add_argument("--expected-head-sha", type=str, required=True, help="Expected SHA256 of classone_heads.pt")
    parser.add_argument("--include-alignbench", action="store_true", help="Run RLCDAlignBench evaluation")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--pause-tokens", type=int, default=8)
    args = parser.parse_args()

    token = get_hf_token()
    print("=" * 80)
    print(f"VALIDATING FRESH DOWNLOADED MODEL: {args.repo_id}")
    print("=" * 80)

    # 1. Download fresh heads directly from HF Hub
    print(f"[*] Downloading fresh classone_heads.pt from {args.repo_id} (force_download=True)...")
    heads_path = hf_hub_download(repo_id=args.repo_id, filename="classone_heads.pt", token=token, force_download=True)
    actual_sha = sha256_file(heads_path)
    print(f"    Downloaded heads path: {heads_path}")
    print(f"    Actual SHA256:   {actual_sha}")
    print(f"    Expected SHA256: {args.expected_head_sha}")
    assert actual_sha == args.expected_head_sha, f"SHA256 mismatch! Got {actual_sha}, expected {args.expected_head_sha}"
    print("    [✓] SHA256 Hash Verification: MATCH CONFIRMED!")

    # 2. Load tokenizer and model from HF Hub
    print(f"[*] Loading tokenizer from {args.repo_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.repo_id, token=token)
    builder = ClassOnePromptBuilder(tokenizer, default_pause_tokens=args.pause_tokens)
    print(f"    Tokenizer length: {len(tokenizer)}")

    print(f"[*] Loading ClassOne model backbone directly from HF Hub: {args.repo_id} (4-bit NF4 on {args.device})...")
    model = ClassOneModel.from_backbone(
        base_model_name_or_path=args.repo_id,
        tokenizer=tokenizer,
        device=args.device,
        torch_dtype=torch.bfloat16,
        quantization="4bit",
    )
    heads = torch.load(heads_path, map_location=args.device)
    model.noul_head.load_state_dict(heads["noul_head"], strict=False)
    model.choice_head.load_state_dict(heads["choice_head"], strict=False)
    model.score_head.load_state_dict(heads["score_head"], strict=False)
    model.eval()
    print("[✓] Model and decision heads ready for evaluation.\n")

    # 3. Run benchmarks
    results = {
        "repo_id": args.repo_id,
        "head_sha256": actual_sha,
    }

    # JevBench
    t0 = time.perf_counter()
    results["jevbench"] = evaluate_jevbench(model, builder, pause_tokens=args.pause_tokens)
    jev_time = time.perf_counter() - t0

    # Decision Index
    t0 = time.perf_counter()
    results["decision_index"] = evaluate_decision_index(model, builder, pause_tokens=args.pause_tokens)
    di_time = time.perf_counter() - t0

    # AlignBench (optional)
    if args.include_alignbench:
        t0 = time.perf_counter()
        results["rlcd_alignbench"] = evaluate_rlcd_alignbench(model, builder, pause_tokens=args.pause_tokens)
        align_time = time.perf_counter() - t0

    # Cleanup GPU memory
    del model
    torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print(f"VALIDATION SUMMARY FOR {args.repo_id}")
    print("=" * 80)
    print(f"Head SHA256:           {actual_sha} [MATCHED]")
    print(f"JevBench Overall:      {results['jevbench']['overall_accuracy']:.1f}% ({results['jevbench']['total_correct']}/{results['jevbench']['total_tasks']}) in {jev_time:.1f}s")
    for tier, tr in results['jevbench']['tiers'].items():
        print(f"  - {tier.capitalize():8s} Tier: {tr['accuracy']:.1f}% ({tr['correct']}/{tr['total']})")
    print(f"HF Decision Index:     {results['decision_index']['overall_decision_index']:.1f}% ({results['decision_index']['total_correct']}/{results['decision_index']['total_tasks']}) in {di_time:.1f}s")
    if args.include_alignbench:
        print(f"RLCDAlignBench AUROC:  {results['rlcd_alignbench']['overall_auroc']:.3f}")
        print(f"RLCDAlignBench BalAcc: {results['rlcd_alignbench']['balanced_accuracy']:.1f}% in {align_time:.1f}s")
    print("=" * 80)

    # Save validation output
    out_name = args.repo_id.split("/")[-1] + "_live_hf_validation.json"
    os.makedirs("benchmarks/results/live_hf_validation", exist_ok=True)
    out_path = os.path.join("benchmarks", "results", "live_hf_validation", out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[✓] Saved validation certificate to {out_path}")


if __name__ == "__main__":
    main()
