#!/usr/bin/env python3
"""Evaluation Runner for NVIDIA When2Call Tool Decision Benchmark (nvidia/When2Call).

Evaluates ClassOne on tool-calling decision making across four discrete execution actions:
1. 'direct': Answer directly without executing tools.
2. 'tool_call': Execute a specific tool call.
3. 'request_for_info': Ask follow-up question to clarify missing parameters.
4. 'cannot_answer': Prompt cannot be answered with available tools.

Measures:
- Overall Categorical Accuracy & Per-Action Class Accuracy
- Expected Calibration Error (ECE) and Normalized Confidence
- Single-Pass Forward Latency (median p50, p95)

Usage:
    python scripts/eval_when2call.py --model devops-thiago/classone-qwen3.5-9b --limit 100
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from classone.modeling.loss import expected_calibration_error
from classone.modeling.modeling_classone import ClassOneModel
from classone.schemas import ChoiceQuestion
from classone.tokenizer import ClassOnePromptBuilder

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

WHEN2CALL_INSTRUCTIONS = "Which response should the assistant give to the user's question, given the available tools?"


def get_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if not token and os.path.exists(".env"):
        with open(".env", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("HF_TOKEN="):
                    token = line.strip().split("=", 1)[1].strip().strip("'\"")
                    break
    return token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ClassOne on NVIDIA When2Call.")
    parser.add_argument(
        "--model",
        type=str,
        default="devops-thiago/classone-qwen3.5-9b",
        help="Model identifier or local checkpoint path.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Execution device."
    )
    parser.add_argument("--limit", type=int, default=100, help="Number of samples to evaluate (0 for full split).")
    parser.add_argument("--heads", type=str, default=None, help="Path to classone_heads.pt file (optional).")
    parser.add_argument("--pause-tokens", type=int, default=8, help="Number of latent pause tokens.")
    parser.add_argument("--quant", type=str, default=None, help="Backbone quantization ('4bit', 'none').")
    parser.add_argument("--no-quant", action="store_true", help="Disable 4-bit quantization (use native BF16/FP16).")
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results/when2call_results.json",
        help="Path to save evaluation metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 80)
    print("NVIDIA WHEN2CALL TOOL-ROUTING BENCHMARK (nvidia/When2Call)")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Pause Tokens: {args.pause_tokens}")
    if args.heads:
        print(f"Heads: {args.heads}")
    print("=" * 80)

    token = get_hf_token()

    # Load dataset with deterministic stratified sampling (seed=20260919 per Decision Index v0.4.1)
    # Note: nvidia/When2Call mcq split is ordered by class; sequential slicing without shuffle only samples 'cannot_answer'.
    print("[*] Loading nvidia/When2Call dataset (split='mcq')...")
    ds = load_dataset("nvidia/When2Call", split="mcq")
    total_available = len(ds)
    indices = list(range(total_available))
    random.Random(20260919).shuffle(indices)
    limit = args.limit if args.limit > 0 else total_available
    selected_indices = indices[: min(limit, total_available)]
    samples = ds.select(selected_indices)
    print(f"    Evaluating {len(samples)} deterministically stratified samples (out of {total_available} total)...")

    # Load model and tokenizer
    print(f"[*] Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token)
    pause_cnt = 0 if "2b" in args.model.lower() else args.pause_tokens
    builder = ClassOnePromptBuilder(tokenizer, default_pause_tokens=pause_cnt)

    is_large = any(k in args.model.lower() for k in ["e4b", "12b", "9b", "4b"])
    if args.no_quant or (args.quant and args.quant.lower() in ("none", "false", "no", "bf16", "fp16")):
        quant = None
    elif args.quant:
        quant = args.quant
    elif is_large and "cuda" in args.device:
        quant = "4bit"
    else:
        quant = None

    print(f"[*] Loading model backbone from {args.model}...")
    model = ClassOneModel.from_backbone(
        base_model_name_or_path=args.model,
        tokenizer=tokenizer,
        device=args.device,
        torch_dtype=torch.bfloat16 if "cuda" in args.device else torch.float32,
        quantization=quant,
    )
    heads_path = args.heads
    if heads_path:
        if not os.path.exists(heads_path):
            raise FileNotFoundError(f"Explicitly specified --heads file not found: {heads_path}")
    else:
        local_cand = os.path.join(args.model, "classone_heads.pt")
        if os.path.exists(local_cand):
            heads_path = local_cand
        elif "/" in args.model:
            try:
                from huggingface_hub import hf_hub_download

                print(f"[*] Downloading classone_heads.pt from {args.model}...")
                heads_path = hf_hub_download(repo_id=args.model, filename="classone_heads.pt", token=token)
            except Exception as e:
                print(f"[!] Warning: Could not download classone_heads.pt: {e}")

    if heads_path and os.path.exists(heads_path):
        lora_dir = os.path.join(os.path.dirname(heads_path), "lora_backbone")
        if os.path.exists(os.path.join(lora_dir, "adapter_config.json")):
            from peft import PeftModel

            print(f"[*] Attaching local LoRA adapter from {lora_dir}...")
            model.backbone = PeftModel.from_pretrained(model.backbone, lora_dir)
        heads_data = torch.load(heads_path, map_location=args.device)
        model.noul_head.load_state_dict(heads_data["noul_head"], strict=False)
        model.choice_head.load_state_dict(heads_data["choice_head"], strict=False)
        model.score_head.load_state_dict(heads_data["score_head"], strict=False)
        print(f"[✓] Loaded custom heads from {heads_path}")
    else:
        print("[!] WARNING: No custom heads loaded; using initialized head weights.")
    model.eval()

    latencies = []
    y_true = []
    y_pred = []
    confidences = []
    per_class_total: dict[str, int] = {}
    per_class_correct: dict[str, int] = {}

    print(f"[*] Starting inference on {len(samples)} samples...")
    for idx, row in enumerate(samples):
        query = row.get("question", "")
        tools_list = row.get("tools", [])
        parsed_tools = []
        for t in tools_list if isinstance(tools_list, list) else []:
            try:
                parsed_tools.append(json.loads(t) if isinstance(t, str) else t)
            except Exception:
                parsed_tools.append(str(t))
        correct_answer = row.get("correct_answer", "")
        answers = row.get("answers", {})

        state_obj = {"tools": parsed_tools, "question": query}
        state_text = json.dumps(state_obj, ensure_ascii=False)

        criteria = {
            "direct": answers.get("direct", "Provide direct answer without calling external tools."),
            "tool_call": answers.get("tool_call", "Invoke a tool call to fulfill the request."),
            "request_for_info": answers.get("request_for_info", "Request missing required parameters from the user."),
            "cannot_answer": answers.get("cannot_answer", "The request cannot be satisfied with the available tools."),
        }

        cq = ChoiceQuestion(
            instructions=WHEN2CALL_INSTRUCTIONS,
            criteria=criteria,
        )

        packed = builder.pack(state_text[:6000], {"q": cq}, num_pause_tokens=pause_cnt)

        t0 = time.perf_counter()
        with torch.no_grad():
            res = model.evaluate_packed(packed)["q"]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency_ms)

        pred_choice = res.choice
        conf = res.confidence

        is_correct = pred_choice == correct_answer
        y_true.append(correct_answer)
        y_pred.append(pred_choice)
        confidences.append(conf)

        per_class_total[correct_answer] = per_class_total.get(correct_answer, 0) + 1
        if is_correct:
            per_class_correct[correct_answer] = per_class_correct.get(correct_answer, 0) + 1

        if (idx + 1) % 25 == 0 or (idx + 1) == len(samples):
            current_acc = sum(yt == yp for yt, yp in zip(y_true, y_pred)) / len(y_true) * 100.0
            print(
                f"  [{idx + 1:4d}/{len(samples)}] Accuracy: {current_acc:5.1f}% | Median Latency: {np.median(latencies):.1f} ms"
            )

    # Aggregate metrics
    overall_acc = sum(yt == yp for yt, yp in zip(y_true, y_pred)) / len(y_true) * 100.0
    p50_latency = float(np.median(latencies))
    p95_latency = float(np.percentile(latencies, 95))

    acc_binary = [1.0 if yt == yp else 0.0 for yt, yp in zip(y_true, y_pred)]
    ece_val = expected_calibration_error(
        torch.tensor(confidences, dtype=torch.float32), torch.tensor(acc_binary, dtype=torch.float32)
    )

    per_class_acc = {cls: (per_class_correct.get(cls, 0) / per_class_total[cls] * 100.0) for cls in per_class_total}

    results_payload = {
        "benchmark": "NVIDIA When2Call (mcq)",
        "model": args.model,
        "sample_count": len(samples),
        "overall_accuracy_pct": round(overall_acc, 2),
        "ece": round(float(ece_val), 4),
        "latency_p50_ms": round(p50_latency, 2),
        "latency_p95_ms": round(p95_latency, 2),
        "per_class_accuracy_pct": per_class_acc,
        "per_class_counts": per_class_total,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    all_results = {}
    if os.path.exists(args.output):
        try:
            with open(args.output, encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict) and "models" in loaded:
                    all_results = loaded
                elif isinstance(loaded, dict) and "benchmark" in loaded:
                    prev_model = loaded.get("model", "previous")
                    all_results = {"benchmark": "NVIDIA When2Call (mcq)", "models": {prev_model: loaded}}
        except Exception:
            pass

    if "models" not in all_results:
        all_results = {"benchmark": "NVIDIA When2Call (mcq)", "models": {}}
    all_results["models"][args.model] = results_payload

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print("FINAL BENCHMARK SUMMARY (NVIDIA When2Call)")
    print(f"Overall Accuracy: {overall_acc:.2f}%")
    print(f"Calibration ECE:  {ece_val:.4f}")
    print(f"Latency (p50):    {p50_latency:.1f} ms")
    print(f"Latency (p95):    {p95_latency:.1f} ms")
    for cls, acc in sorted(per_class_acc.items()):
        print(f"  - {cls:18s}: {acc:5.1f}% ({per_class_correct.get(cls, 0)}/{per_class_total[cls]})")
    print(f"Saved results to {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
