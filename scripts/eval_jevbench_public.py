#!/usr/bin/env python3
"""Evaluates ClassOne on the public JevBench easy tier tasks.

Source: https://raw.githubusercontent.com/fstandhartinger/jevbench/main/datasets/public/easy.jsonl
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
from classone.schemas import ChoiceQuestion
from classone.tokenizer import ClassOnePromptBuilder

REPO_ID = "devops-thiago/classone-gemma4-e2b"
DATA_URL = "https://raw.githubusercontent.com/fstandhartinger/jevbench/main/datasets/public/easy.jsonl"


def fetch_easy_tasks():
    with urllib.request.urlopen(DATA_URL, timeout=15) as resp:
        lines = resp.read().decode("utf-8").strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_path = sys.argv[1] if len(sys.argv) > 1 else REPO_ID
    heads_path = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/classone_gemma4_e2b/classone_heads.pt"
    if not os.path.exists(heads_path):
        heads_path = hf_hub_download(REPO_ID, "classone_heads.pt")

    print(f"Loading {model_path} on {device} (heads from {heads_path})...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    builder = ClassOnePromptBuilder(tokenizer)

    model = ClassOneModel.from_backbone(
        base_model_name_or_path=model_path,
        tokenizer=tokenizer,
        device=device,
        torch_dtype=torch.float16,
    )
    heads = torch.load(heads_path, map_location=device)
    model.noul_head.load_state_dict(heads["noul_head"], strict=False)
    model.choice_head.load_state_dict(heads["choice_head"], strict=False)
    model.score_head.load_state_dict(heads["score_head"], strict=False)
    model.eval()

    tasks = fetch_easy_tasks()
    print(f"Fetched {len(tasks)} public easy-tier tasks from JevBench.\n")

    correct = 0
    total = 0
    latencies = []
    all_top_probs = []
    all_accuracies = []

    for task in tasks:
        state = task["state"]
        q_data = task["question"]
        expected = task["expected"]

        # Only evaluate choice questions in this tier
        if q_data.get("type") != "choice":
            continue

        q = ChoiceQuestion(
            instructions=q_data["instructions"],
            criteria=q_data["criteria"],
        )
        packed = builder.pack(state, {"q": q})

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            res = model.evaluate_packed(packed)["q"]
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

        pred = res.choice
        is_correct = 1.0 if pred == expected else 0.0
        correct += int(is_correct)
        total += 1

        top_prob = max(res.probabilities.values())
        all_top_probs.append(top_prob)
        all_accuracies.append(is_correct)

    accuracy = (correct / total) * 100.0 if total > 0 else 0.0
    mean_lat = float(np.mean(latencies))
    p50_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))

    probs_tensor = torch.tensor(all_top_probs, dtype=torch.float32)
    acc_tensor = torch.tensor(all_accuracies, dtype=torch.float32)
    ece = expected_calibration_error(probs_tensor, acc_tensor, n_bins=10)

    print("=" * 60)
    print("      JEVBENCH PUBLIC EASY TIER EVALUATION RESULTS")
    print("=" * 60)
    print(f"Tasks Evaluated      : {total}")
    print(f"Accuracy             : {correct}/{total} ({accuracy:.1f}%)")
    print(f"Expected Calib Error : {ece:.4f}")
    print(f"Mean Latency         : {mean_lat:.2f} ms")
    print(f"Median (p50) Latency : {p50_lat:.2f} ms")
    print(f"p95 Latency          : {p95_lat:.2f} ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
