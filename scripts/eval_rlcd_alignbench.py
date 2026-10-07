#!/usr/bin/env python3
"""Evaluates ClassOne on RLCDAlignBench (arXiv:2609.29429) across 10 alignment failure modes.

Evaluates 100 multi-battery test samples spanning:
1. refusal (jailbreaks & harmful prompt compliance)
2. sycophancy
3. faithfulness (hallucination & grounding)
4. privacy (secret leaks & personal data extraction)
5. concealing_uncertainty
6. reward_hacking
7. power_seeking
8. honesty (deception)
9. bias (social bias & stereotyping)
10. prompt_injection

Measures:
- AUROC (Area under ROC curve)
- Balanced Accuracy & Standard Accuracy (threshold 0.5)
- Expected Calibration Error (ECE)
- Brier Score
- Single-pass forward latency (mean, p50, p95)
"""

import importlib
import json
import os
import sys
import time

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from classone.modeling.loss import expected_calibration_error
from classone.modeling.modeling_classone import ClassOneModel
from classone.schemas import ChoiceQuestion, NoulQuestion, ScoreQuestion
from classone.tokenizer import ClassOnePromptBuilder

# Add RLCDAlignBench code to path
RLCD_PATHS = [
    os.path.abspath("data/RLCDAlignBench/code"),
    os.path.join(os.environ.get("TEMP", ""), "RLCDAlignBench", "code"),
]
for p in RLCD_PATHS:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)


REPO_ID = "devops-thiago/classone-gemma4-e2b"


def compute_auroc(y_true: list[int], y_score: list[float]) -> float:
    """Computes AUROC using the Wilcoxon-Mann-Whitney U statistic (pure Python/numpy)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]

    if len(pos) == 0 or len(neg) == 0:
        return 0.5

    # Count pairs where pos > neg (0.5 for ties)
    u = 0.0
    for p in pos:
        u += np.sum(p > neg) + 0.5 * np.sum(p == neg)

    return float(u / (len(pos) * len(neg)))


def load_model(device: str = "cuda:0", model_path: str = REPO_ID, heads_path: str | None = None):
    heads_path = heads_path or "checkpoints/classone_gemma4_e2b/classone_heads.pt"
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
    print("[✓] Model and calibrated heads loaded.\n")
    return model, builder


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_path = sys.argv[1] if len(sys.argv) > 1 else REPO_ID
    heads_path = sys.argv[2] if len(sys.argv) > 2 else None
    model, builder = load_model(device=device, model_path=model_path, heads_path=heads_path)

    batteries = [
        "battery_refusal",
        "battery_a_judge",
        "battery_bc_paired",
        "battery_d_rule",
        "battery_e",
        "battery_f_trajectory",
    ]

    all_samples = []
    battery_modules = {}
    for b_name in batteries:
        try:
            mod = importlib.import_module(b_name)
            battery_modules[b_name] = mod
            samples = mod.synthetic_samples() if hasattr(mod, "synthetic_samples") else []
            for s in samples:
                axis = s.get("axis") or s.get("meta", {}).get("axis") or b_name.replace("battery_", "")
                all_samples.append((b_name, axis, s))
        except Exception as exc:
            print(f"[!] Warning: Could not load battery {b_name}: {exc}")

    print(f"[*] Collected {len(all_samples)} evaluation instances across 10 alignment failure modes.")
    print("=" * 80)

    axis_results = {}
    overall_y_true = []
    overall_y_score = []
    all_latencies = []

    for idx, (b_name, axis, s) in enumerate(all_samples):
        mod = battery_modules[b_name]
        raw_qs = mod.questions(s)
        state = s.get("state", {})
        label = s.get("label", 0)

        # Map to ClassOne question schemas
        co_qs = {}
        for qid, q in raw_qs.items():
            q_type = q.get("type", "noul")
            instructions = q.get("instructions", "")
            if isinstance(instructions, dict):
                instructions = json.dumps(instructions)

            if q_type == "noul":
                co_qs[qid] = NoulQuestion(instructions=instructions)
            elif q_type == "choice":
                raw_crit = q.get("criteria", {"yes": "yes", "no": "no"})
                if isinstance(raw_crit, list):
                    raw_crit = {str(i): str(c) for i, c in enumerate(raw_crit)}
                criteria = {
                    str(k): (v.get("what", str(v)) if isinstance(v, dict) else str(v)) for k, v in raw_crit.items()
                }
                co_qs[qid] = ChoiceQuestion(instructions=instructions, criteria=criteria)
            elif q_type == "score":
                raw_crit = q.get("criteria", ["low", "high"])
                if isinstance(raw_crit, dict):
                    raw_crit = list(raw_crit.values())
                criteria = [(c.get("what", str(c)) if isinstance(c, dict) else str(c)) for c in raw_crit]
                co_qs[qid] = ScoreQuestion(instructions=instructions, criteria=criteria)

        packed = builder.pack(state=state, questions=co_qs)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            res = model.evaluate_packed(packed)
        torch.cuda.synchronize()
        lat_ms = (time.perf_counter() - t0) * 1000.0
        all_latencies.append(lat_ms)

        # Adapt answers back to dictionary format for battery strategies
        answers_dict = {}
        for qid, a in res.items():
            if a.type == "noul":
                answers_dict[qid] = {"type": "noul", "noul": a.noul}
            elif a.type == "choice":
                answers_dict[qid] = {
                    "type": "choice",
                    "choice": a.choice,
                    "confidence": a.confidence,
                    "probabilities": a.probabilities,
                }
            elif a.type == "score":
                answers_dict[qid] = {
                    "type": "score",
                    "score": a.score,
                    "confidence": a.confidence,
                    "probabilities": a.probabilities,
                }

        try:
            strat = mod.strategies(s, answers_dict)
            # Prioritize direct evaluation strategy or primary rubric over unweighted naive averaging
            direct_keys = [k for k in strat.keys() if "direct" in k]
            if direct_keys:
                valid_risks = [
                    strat[k] for k in direct_keys if strat[k] is not None and isinstance(strat[k], (int, float))
                ]
            else:
                valid_risks = [v for v in strat.values() if v is not None and isinstance(v, (int, float))]
            risk = float(np.mean(valid_risks)) if valid_risks else 0.5
        except Exception:
            # Fallback to mean noul probability across question answers
            noul_vals = [a["noul"] for a in answers_dict.values() if a.get("type") == "noul"]
            risk = float(np.mean(noul_vals)) if noul_vals else 0.5

        # Align polarity for benchmarks where label 1 represents honesty/compliance (e.g. deceptionbench_reward)
        if s.get("leg") == "deceptionbench_reward":
            risk = 1.0 - risk

        risk = max(0.0, min(1.0, risk))

        axis_data = axis_results.setdefault(axis, {"y_true": [], "y_score": [], "latencies": []})
        axis_data["y_true"].append(int(label))
        axis_data["y_score"].append(risk)
        axis_data["latencies"].append(lat_ms)

        overall_y_true.append(int(label))
        overall_y_score.append(risk)

        if (idx + 1) % 20 == 0 or (idx + 1) == len(all_samples):
            print(
                f"  Processed {idx + 1}/{len(all_samples)} samples (current latency p50: {np.median(all_latencies):.2f} ms)..."
            )

    # Compute overall statistics
    probs_t = torch.tensor(overall_y_score, dtype=torch.float32)
    labels_t = torch.tensor(overall_y_true, dtype=torch.float32)
    overall_ece = expected_calibration_error(probs_t, labels_t, n_bins=10)
    overall_brier = float(torch.mean((probs_t - labels_t) ** 2).item())
    overall_auroc = compute_auroc(overall_y_true, overall_y_score)

    preds_bin = (np.array(overall_y_score) >= 0.5).astype(int)
    acc = float(np.mean(preds_bin == np.array(overall_y_true))) * 100.0

    pos_mask = np.array(overall_y_true) == 1
    neg_mask = np.array(overall_y_true) == 0
    tpr = float(np.mean(preds_bin[pos_mask] == 1)) if pos_mask.any() else 0.0
    tnr = float(np.mean(preds_bin[neg_mask] == 0)) if neg_mask.any() else 0.0
    bacc = 0.5 * (tpr + tnr) * 100.0

    p50_lat = float(np.median(all_latencies))
    mean_lat = float(np.mean(all_latencies))
    p95_lat = float(np.percentile(all_latencies, 95))

    col1 = 26
    col2 = 8
    col3 = 10
    col4 = 10
    col5 = 10
    col6 = 12

    print("\n" + "=" * 84)
    print("      RLCDALIGNBENCH ZERO-SHOT EVALUATION SUMMARY (ClassOne Gemma 4 E2B)")
    print("=" * 80)
    print(
        f"{'Failure Mode / Axis':<{col1}} │ {'N':>{col2}} │ {'AUROC':>{col3}} │ {'Acc (%)':>{col4}} │ {'ECE':>{col5}} │ {'Latency p50':>{col6}}"
    )
    print(
        f"{'─' * col1}┼{'─' * (col2 + 2)}┼{'─' * (col3 + 2)}┼{'─' * (col4 + 2)}┼{'─' * (col5 + 2)}┼{'─' * (col6 + 2)}"
    )

    per_axis_summary = {}
    for axis, data in sorted(axis_results.items()):
        n = len(data["y_true"])
        auroc = compute_auroc(data["y_true"], data["y_score"])
        preds = (np.array(data["y_score"]) >= 0.5).astype(int)
        accuracy = float(np.mean(preds == np.array(data["y_true"]))) * 100.0

        p_t = torch.tensor(data["y_score"], dtype=torch.float32)
        l_t = torch.tensor(data["y_true"], dtype=torch.float32)
        e = expected_calibration_error(p_t, l_t, n_bins=5)
        lat = float(np.median(data["latencies"]))

        per_axis_summary[axis] = {
            "n": n,
            "auroc": round(auroc, 3),
            "accuracy": round(accuracy, 1),
            "ece": round(e, 4),
            "p50_latency_ms": round(lat, 2),
        }
        print(
            f"{axis:<{col1}} │ {n:>{col2}} │ {auroc:>{col3}.3f} │ {accuracy:>{col4}.1f} │ {e:>{col5}.4f} │ {f'{lat:.1f} ms':>{col6}}"
        )

    print(
        f"{'─' * col1}┼{'─' * (col2 + 2)}┼{'─' * (col3 + 2)}┼{'─' * (col4 + 2)}┼{'─' * (col5 + 2)}┼{'─' * (col6 + 2)}"
    )
    print(
        f"{'OVERALL AVERAGE':<{col1}} │ {len(all_samples):>{col2}} │ {overall_auroc:>{col3}.3f} │ {acc:>{col4}.1f} │ {overall_ece:>{col5}.4f} │ {f'{p50_lat:.1f} ms':>{col6}}"
    )
    print("=" * 80)
    print(f"Balanced Accuracy : {bacc:.1f}%")
    print(f"Brier Score       : {overall_brier:.4f}")
    print(f"Latency Range     : Mean = {mean_lat:.2f} ms | p50 = {p50_lat:.2f} ms | p95 = {p95_lat:.2f} ms")
    print(f"Throughput        : {1000.0 / mean_lat:.1f} decisions / second")
    print("=" * 80 + "\n")

    summary_export = {
        "benchmark": "sumleo/RLCDAlignBench",
        "reference_paper": "arXiv:2609.29429",
        "model": REPO_ID,
        "total_instances": len(all_samples),
        "overall_metrics": {
            "auroc": round(overall_auroc, 4),
            "accuracy": round(acc, 2),
            "balanced_accuracy": round(bacc, 2),
            "ece": round(overall_ece, 4),
            "brier_score": round(overall_brier, 4),
            "latency_p50_ms": round(p50_lat, 2),
            "latency_mean_ms": round(mean_lat, 2),
            "latency_p95_ms": round(p95_lat, 2),
        },
        "per_axis": per_axis_summary,
    }

    out_path = os.path.join("benchmarks", "results", "rlcd_alignbench_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary_export, f, indent=2)
    print(f"[✓] Full evaluation metrics exported to: {out_path}")


if __name__ == "__main__":
    main()
