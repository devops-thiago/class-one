#!/usr/bin/env python3
"""Benchmark script evaluating the published Hugging Face model on dual-GPU CUDA.

Loads the published merged standalone model (devops-thiago/classone-gemma4-e2b) on cuda:0
and the autoregressive Gemma baseline on cuda:1 simultaneously.

Usage:
    python scripts/benchmark_published.py --iterations 30 --warmup 5
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from classone.modeling.modeling_classone import ClassOneModel
from classone.schemas import ChoiceQuestion, NoulQuestion, ScoreQuestion
from classone.tokenizer import ClassOnePromptBuilder

DEFAULT_MODEL_ID = "devops-thiago/classone-gemma4-e2b"
DEFAULT_BASE_MODEL_ID = "google/gemma-4-e2b-it"


def parse_args():
    parser = argparse.ArgumentParser(description="ClassOne Published Model Dual-GPU Benchmark")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ID, help="HF Hub model ID")
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL_ID,
        help="Base autoregressive model ID for comparison",
    )
    parser.add_argument("--iterations", type=int, default=30, help="Benchmark timing iterations")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--classone-gpu", type=int, default=0, help="GPU index for ClassOne")
    parser.add_argument("--ar-gpu", type=int, default=1, help="GPU index for AR baseline")
    parser.add_argument(
        "--output-json",
        type=str,
        default="benchmarks/results/benchmark_published_results.json",
        help="Path to output JSON metrics",
    )
    return parser.parse_args()


def ms(val: float) -> str:
    return f"{val:.2f} ms"


def main():
    args = parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    if not torch.cuda.is_available():
        print("[!] CUDA is required for this real-model benchmark.", file=sys.stderr)
        sys.exit(1)

    n_gpus = torch.cuda.device_count()
    co_device = f"cuda:{args.classone_gpu}"
    ar_device = f"cuda:{args.ar_gpu}" if n_gpus > 1 else f"cuda:{args.classone_gpu}"
    ar_gpu_idx = args.ar_gpu if n_gpus > 1 else args.classone_gpu

    print("=" * 72)
    print("   ClassOne Published Model Latency & Throughput Benchmark")
    print(f"   Published Repo : {args.model}")
    print(f"   Base AR Model  : {args.base_model}")
    for i in range(n_gpus):
        vram = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"   GPU {i}          : {torch.cuda.get_device_name(i)} ({vram:.1f} GB)")
    print(f"   Precision      : {args.dtype} | Iterations: {args.iterations} | Warmup: {args.warmup}")
    print("=" * 72)

    # 1. Tokenizer and Prompt Building
    print(f"\n[1/4] Loading published tokenizer from {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    builder = ClassOnePromptBuilder(tokenizer)

    state = {
        "customer": "Jordan M.",
        "transaction_id": 99882,
        "message": "Payment failed but funds were deducted from checking account.",
    }
    questions = {
        "is_fraud": NoulQuestion(instructions="Is this a potential fraud event?"),
        "queue": ChoiceQuestion(
            instructions="Routing queue:",
            criteria={"billing": "Billing and disputes", "support": "Customer support", "tech": "Technical bugs"},
        ),
        "urgency": ScoreQuestion(
            instructions="Urgency level:",
            criteria=["low", "medium", "critical"],
        ),
    }

    packed = builder.pack(state=state, questions=questions)
    seq_len = packed.input_ids.shape[1]
    print(f"      Packed ClassOne prompt: {seq_len} tokens")

    # Autoregressive baseline prompt
    ar_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    ar_prompt = (
        "<start_of_turn>user\n"
        "Given this customer event:\n"
        '{"customer": "Jordan M.", "transaction_id": 99882, "message": "Payment failed but funds were deducted from checking account."}\n\n'
        "Answer in JSON with three fields:\n"
        "  is_fraud (true/false), queue (billing|support|tech), urgency (low|medium|critical).\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
        "```json\n"
    )
    ar_enc = ar_tokenizer(ar_prompt, return_tensors="pt")
    ar_input = ar_enc.input_ids.to(ar_device)
    pad_id = ar_tokenizer.pad_token_id or ar_tokenizer.eos_token_id
    print(f"      AR prompt: {ar_enc.input_ids.shape[1]} tokens")

    # 2. Loading Published ClassOne Model
    print(f"\n[2/4] Loading published ClassOne model onto {co_device} ...")
    torch.cuda.reset_peak_memory_stats(args.classone_gpu)
    t0_load = time.time()
    co_model = ClassOneModel.from_backbone(
        base_model_name_or_path=args.model,
        tokenizer=tokenizer,
        device=co_device,
        torch_dtype=torch_dtype,
    )
    heads_path = hf_hub_download(args.model, "classone_heads.pt")
    heads = torch.load(heads_path, map_location=co_device)
    co_model.noul_head.load_state_dict(heads["noul_head"])
    co_model.choice_head.load_state_dict(heads["choice_head"])
    co_model.score_head.load_state_dict(heads["score_head"])
    co_model.eval()

    co_load_time = time.time() - t0_load
    co_vram_gb = torch.cuda.memory_allocated(args.classone_gpu) / (1024**3)
    co_vram_peak = torch.cuda.max_memory_allocated(args.classone_gpu) / (1024**3)
    print(f"      ClassOne loaded in {co_load_time:.2f}s | VRAM: {co_vram_gb:.2f} GB (Peak: {co_vram_peak:.2f} GB)")

    # 3. Loading Autoregressive Baseline Model
    print(f"\n[3/4] Loading autoregressive Gemma baseline onto {ar_device} ...")
    torch.cuda.reset_peak_memory_stats(ar_gpu_idx)
    t0_load_ar = time.time()
    ar_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch_dtype,
        device_map=ar_device,
    )
    ar_model.eval()
    ar_load_time = time.time() - t0_load_ar
    ar_vram_gb = torch.cuda.memory_allocated(ar_gpu_idx) / (1024**3)
    ar_vram_peak = torch.cuda.max_memory_allocated(ar_gpu_idx) / (1024**3)
    print(f"      AR model loaded in {ar_load_time:.2f}s | VRAM: {ar_vram_gb:.2f} GB (Peak: {ar_vram_peak:.2f} GB)")

    # 4. Executing Benchmark
    print("\n[4/4] Running performance benchmarks...")

    # Phase A: ClassOne Single-Pass
    print(f"      [*] ClassOne warmup ({args.warmup} iterations)...")
    for _ in range(args.warmup):
        with torch.no_grad():
            co_model.evaluate_packed(packed)
        torch.cuda.synchronize(args.classone_gpu)

    print(f"      [*] ClassOne timed runs ({args.iterations} iterations)...")
    co_latencies = []
    for _ in range(args.iterations):
        torch.cuda.synchronize(args.classone_gpu)
        t_start = time.perf_counter()
        with torch.no_grad():
            co_model.evaluate_packed(packed)
        torch.cuda.synchronize(args.classone_gpu)
        co_latencies.append((time.perf_counter() - t_start) * 1000.0)

    co_mean = float(np.mean(co_latencies))
    co_p50 = float(np.percentile(co_latencies, 50))
    co_p90 = float(np.percentile(co_latencies, 90))
    co_p95 = float(np.percentile(co_latencies, 95))
    co_p99 = float(np.percentile(co_latencies, 99))
    co_min = float(np.min(co_latencies))
    co_max = float(np.max(co_latencies))
    co_rps = 1000.0 / co_mean

    # Phase B: Autoregressive Baseline (50 tokens)
    print(f"      [*] AR baseline warmup ({args.warmup} iterations)...")
    for _ in range(args.warmup):
        with torch.no_grad():
            ar_model.generate(ar_input, max_new_tokens=50, do_sample=False, pad_token_id=pad_id)
        torch.cuda.synchronize(ar_gpu_idx)

    print(f"      [*] AR baseline timed runs ({args.iterations} iterations)...")
    ar_latencies = []
    for _ in range(args.iterations):
        torch.cuda.synchronize(ar_gpu_idx)
        t_start = time.perf_counter()
        with torch.no_grad():
            ar_model.generate(ar_input, max_new_tokens=50, do_sample=False, pad_token_id=pad_id)
        torch.cuda.synchronize(ar_gpu_idx)
        ar_latencies.append((time.perf_counter() - t_start) * 1000.0)

    ar_mean = float(np.mean(ar_latencies))
    ar_p50 = float(np.percentile(ar_latencies, 50))
    ar_p90 = float(np.percentile(ar_latencies, 90))
    ar_p95 = float(np.percentile(ar_latencies, 95))
    ar_p99 = float(np.percentile(ar_latencies, 99))
    ar_min = float(np.min(ar_latencies))
    ar_max = float(np.max(ar_latencies))
    ar_rps = 1000.0 / ar_mean

    speedup = ar_mean / co_mean

    # Summary Display
    col1 = 28
    col2 = 22
    col3 = 22
    sep = f"{'─' * col1}┼{'─' * (col2 + 2)}┼{'─' * (col3 + 2)}"

    print("\n" + "=" * 76)
    print("           PUBLISHED MODEL BENCHMARK RESULTS SUMMARY")
    print("=" * 76)
    print(f"{'Metric':<{col1}} │ {'ClassOne (Single-Pass)':>{col2}} │ {'Autoregressive (50t)':>{col3}}")
    print(sep)
    print(f"{'Mean Latency':<{col1}} │ {ms(co_mean):>{col2}} │ {ms(ar_mean):>{col3}}")
    print(f"{'Median (P50)':<{col1}} │ {ms(co_p50):>{col2}} │ {ms(ar_p50):>{col3}}")
    print(f"{'P90 Latency':<{col1}} │ {ms(co_p90):>{col2}} │ {ms(ar_p90):>{col3}}")
    print(f"{'P95 Latency':<{col1}} │ {ms(co_p95):>{col2}} │ {ms(ar_p95):>{col3}}")
    print(f"{'P99 Latency':<{col1}} │ {ms(co_p99):>{col2}} │ {ms(ar_p99):>{col3}}")
    print(f"{'Min Latency':<{col1}} │ {ms(co_min):>{col2}} │ {ms(ar_min):>{col3}}")
    print(f"{'Max Latency':<{col1}} │ {ms(co_max):>{col2}} │ {ms(ar_max):>{col3}}")
    print(f"{'Throughput':<{col1}} │ {f'{co_rps:.1f} req/s':>{col2}} │ {f'{ar_rps:.1f} req/s':>{col3}}")
    print(f"{'Tokens Generated':<{col1}} │ {'0 tokens':>{col2}} │ {'50 tokens':>{col3}}")
    print(f"{'VRAM Allocated':<{col1}} │ {f'{co_vram_gb:.2f} GB':>{col2}} │ {f'{ar_vram_gb:.2f} GB':>{col3}}")
    print(sep)
    print(f"  Speedup: ClassOne is {speedup:.1f}x FASTER than autoregressive generation")
    print("=" * 76 + "\n")

    summary_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "published_model": args.model,
        "base_model": args.base_model,
        "precision": args.dtype,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "hardware": {
            "gpus": [
                {
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "vram_gb": round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2),
                }
                for i in range(n_gpus)
            ],
            "classone_device": co_device,
            "ar_device": ar_device,
        },
        "classone_metrics": {
            "mean_ms": round(co_mean, 2),
            "p50_ms": round(co_p50, 2),
            "p90_ms": round(co_p90, 2),
            "p95_ms": round(co_p95, 2),
            "p99_ms": round(co_p99, 2),
            "min_ms": round(co_min, 2),
            "max_ms": round(co_max, 2),
            "throughput_rps": round(co_rps, 2),
            "vram_allocated_gb": round(co_vram_gb, 2),
            "vram_peak_gb": round(co_vram_peak, 2),
        },
        "ar_baseline_metrics": {
            "mean_ms": round(ar_mean, 2),
            "p50_ms": round(ar_p50, 2),
            "p90_ms": round(ar_p90, 2),
            "p95_ms": round(ar_p95, 2),
            "p99_ms": round(ar_p99, 2),
            "min_ms": round(ar_min, 2),
            "max_ms": round(ar_max, 2),
            "throughput_rps": round(ar_rps, 2),
            "vram_allocated_gb": round(ar_vram_gb, 2),
            "vram_peak_gb": round(ar_vram_peak, 2),
        },
        "speedup_factor": round(speedup, 2),
    }

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(summary_data, f, indent=2)
        print(f"[✓] Detailed results exported to: {args.output_json}")


if __name__ == "__main__":
    main()
