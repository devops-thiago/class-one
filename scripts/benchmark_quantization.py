#!/usr/bin/env python3
"""Benchmark comparing FP16, 8-bit, and 4-bit quantization modes for ClassOne.

Measures:
- VRAM allocated and peak (GB)
- Latency percentiles (mean, p50, p95)
- Throughput (req/s)
"""

import gc
import json
import time

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from classone.modeling.modeling_classone import ClassOneModel
from classone.schemas import ChoiceQuestion, NoulQuestion, ScoreQuestion
from classone.tokenizer import ClassOnePromptBuilder

REPO_ID = "devops-thiago/classone-gemma4-e2b"


def benchmark_mode(mode_name: str, quant_arg: str | None, iterations: int = 20, warmup: int = 3):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
    builder = ClassOnePromptBuilder(tokenizer)

    t0_load = time.time()
    model = ClassOneModel.from_backbone(
        base_model_name_or_path=REPO_ID,
        tokenizer=tokenizer,
        device="cuda:0",
        torch_dtype=torch.float16,
        quantization=quant_arg,
    )
    heads_path = hf_hub_download(REPO_ID, "classone_heads.pt")
    heads = torch.load(heads_path, map_location="cuda:0")
    model.noul_head.load_state_dict(heads["noul_head"])
    model.choice_head.load_state_dict(heads["choice_head"])
    model.score_head.load_state_dict(heads["score_head"])
    model.eval()
    load_time = time.time() - t0_load

    vram_alloc = torch.cuda.memory_allocated() / (1024**3)
    vram_peak = torch.cuda.max_memory_allocated() / (1024**3)

    state = {
        "customer": "Sarah Chen",
        "tier": "enterprise",
        "message": "We experienced an outage on our webhook pipeline. Several webhook deliveries failed. We need an SLA refund credit applied.",
    }
    questions = {
        "refund_request": NoulQuestion(instructions="Is customer asking for a refund?"),
        "routing": ChoiceQuestion(
            instructions="Target team:",
            criteria={"billing": "Billing credits", "tech": "Outages & bugs", "general": "General"},
        ),
        "urgency": ScoreQuestion(
            instructions="Urgency rating:",
            criteria=["low", "normal", "high", "critical"],
        ),
    }
    packed = builder.pack(state, questions)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            model.evaluate_packed(packed)
        torch.cuda.synchronize()

    latencies = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            res = model.evaluate_packed(packed)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = float(np.mean(latencies))
    p50_ms = float(np.median(latencies))
    p95_ms = float(np.percentile(latencies, 95))
    min_ms = float(np.min(latencies))
    max_ms = float(np.max(latencies))
    rps = 1000.0 / mean_ms

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "mode": mode_name,
        "load_time_s": round(load_time, 2),
        "vram_alloc_gb": round(vram_alloc, 2),
        "vram_peak_gb": round(vram_peak, 2),
        "mean_ms": round(mean_ms, 2),
        "p50_ms": round(p50_ms, 2),
        "p95_ms": round(p95_ms, 2),
        "min_ms": round(min_ms, 2),
        "max_ms": round(max_ms, 2),
        "rps": round(rps, 1),
        "sample_output": {
            "refund": res["refund_request"].noul,
            "routing": res["routing"].choice,
            "routing_conf": res["routing"].confidence,
            "urgency": res["urgency"].score,
        },
    }


def main():
    print("=" * 76)
    print("        ClassOne Quantization Benchmark: FP16 vs 8-bit vs 4-bit")
    print("=" * 76)

    modes = [
        ("FP16 (Baseline)", None),
        ("8-bit (INT8)", "8bit"),
        ("4-bit (NF4)", "4bit"),
    ]

    results = []
    for mode_name, quant_arg in modes:
        print(f"\n[*] Evaluating {mode_name}...")
        res = benchmark_mode(mode_name, quant_arg, iterations=20, warmup=3)
        results.append(res)
        print(f"    VRAM: {res['vram_alloc_gb']:.2f} GB | Latency p50: {res['p50_ms']:.2f} ms | RPS: {res['rps']:.1f}")

    col1 = 20
    col2 = 14
    col3 = 14
    col4 = 14
    sep = f"{'─' * col1}┼{'─' * (col2 + 2)}┼{'─' * (col3 + 2)}┼{'─' * (col4 + 2)}"

    print("\n" + "=" * 76)
    print("                     QUANTIZATION BENCHMARK SUMMARY")
    print("=" * 76)
    print(f"{'Metric':<{col1}} │ {'FP16 (Baseline)':>{col2}} │ {'8-bit (INT8)':>{col3}} │ {'4-bit (NF4)':>{col4}}")
    print(sep)

    fp16 = results[0]
    int8 = results[1]
    nf4 = results[2]

    vram_fp16 = f"{fp16['vram_alloc_gb']:.2f} GB"
    vram_int8 = f"{int8['vram_alloc_gb']:.2f} GB"
    vram_nf4 = f"{nf4['vram_alloc_gb']:.2f} GB"

    peak_fp16 = f"{fp16['vram_peak_gb']:.2f} GB"
    peak_int8 = f"{int8['vram_peak_gb']:.2f} GB"
    peak_nf4 = f"{nf4['vram_peak_gb']:.2f} GB"

    savings_int8 = f"-{(1 - int8['vram_alloc_gb'] / fp16['vram_alloc_gb']) * 100:.1f}%"
    savings_nf4 = f"-{(1 - nf4['vram_alloc_gb'] / fp16['vram_alloc_gb']) * 100:.1f}%"

    p50_fp16 = f"{fp16['p50_ms']:.2f} ms"
    p50_int8 = f"{int8['p50_ms']:.2f} ms"
    p50_nf4 = f"{nf4['p50_ms']:.2f} ms"

    mean_fp16 = f"{fp16['mean_ms']:.2f} ms"
    mean_int8 = f"{int8['mean_ms']:.2f} ms"
    mean_nf4 = f"{nf4['mean_ms']:.2f} ms"

    p95_fp16 = f"{fp16['p95_ms']:.2f} ms"
    p95_int8 = f"{int8['p95_ms']:.2f} ms"
    p95_nf4 = f"{nf4['p95_ms']:.2f} ms"

    rps_fp16 = f"{fp16['rps']:.1f} req/s"
    rps_int8 = f"{int8['rps']:.1f} req/s"
    rps_nf4 = f"{nf4['rps']:.1f} req/s"

    print(f"{'VRAM Allocated':<{col1}} │ {vram_fp16:>{col2}} │ {vram_int8:>{col3}} │ {vram_nf4:>{col4}}")
    print(f"{'VRAM Peak':<{col1}} │ {peak_fp16:>{col2}} │ {peak_int8:>{col3}} │ {peak_nf4:>{col4}}")
    print(f"{'VRAM Savings':<{col1}} │ {'0%':>{col2}} │ {savings_int8:>{col3}} │ {savings_nf4:>{col4}}")
    print(f"{'Median Latency':<{col1}} │ {p50_fp16:>{col2}} │ {p50_int8:>{col3}} │ {p50_nf4:>{col4}}")
    print(f"{'Mean Latency':<{col1}} │ {mean_fp16:>{col2}} │ {mean_int8:>{col3}} │ {mean_nf4:>{col4}}")
    print(f"{'p95 Latency':<{col1}} │ {p95_fp16:>{col2}} │ {p95_int8:>{col3}} │ {p95_nf4:>{col4}}")
    print(f"{'Throughput':<{col1}} │ {rps_fp16:>{col2}} │ {rps_int8:>{col3}} │ {rps_nf4:>{col4}}")
    print(f"{'Min GPU Required':<{col1}} │ {'12 GB GPU':>{col2}} │ {'8 GB GPU':>{col3}} │ {'6–8 GB GPU':>{col4}}")
    print("=" * 76 + "\n")

    with open("benchmark_quantization_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[✓] Detailed benchmark exported to: benchmark_quantization_results.json")


if __name__ == "__main__":
    main()
