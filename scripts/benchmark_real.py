#!/usr/bin/env python3
"""Real-model latency benchmark for ClassOne on dual-GPU CUDA.

Loads ClassOne (backbone + LoRA + decision heads) on cuda:0 and the autoregressive
CausalLM baseline on cuda:1 *simultaneously*, using both RTX 5060 Ti GPUs (~17 GB each).

Usage:
    python scripts/benchmark_real.py \
        --checkpoint ./checkpoints/classone_gemma4_e2b \
        --model google/gemma-4-e2b-it \
        --iterations 30 --warmup 5
"""

import argparse
import json
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from classone.modeling.modeling_classone import ClassOneModel
from classone.schemas import ChoiceQuestion, NoulQuestion, ScoreQuestion
from classone.tokenizer import ClassOnePromptBuilder
from classone.train.trainer import ClassOneTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="ClassOne Dual-GPU Real-Model Benchmark")
    parser.add_argument("--model", type=str, default="google/gemma-4-e2b-it")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/classone_gemma4_e2b")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--classone-gpu", type=int, default=0, help="GPU index for the ClassOne model (default: 0)")
    parser.add_argument("--ar-gpu", type=int, default=1, help="GPU index for the AR baseline model (default: 1)")
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def ms(t: float) -> str:
    return f"{t:.2f} ms"


def main():
    args = parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    if not torch.cuda.is_available():
        print("[!] CUDA not available.", file=sys.stderr)
        sys.exit(1)

    n_gpus = torch.cuda.device_count()

    # ------------------------------------------------------------------ #
    print("=" * 68)
    print("  ClassOne — Dual-GPU Real-Model Latency Benchmark")
    for i in range(n_gpus):
        vram = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}  ({vram:.1f} GB)")
    print(f"  dtype: {args.dtype}   iterations: {args.iterations}   warmup: {args.warmup}")
    print("=" * 68)

    co_device = f"cuda:{args.classone_gpu}"
    ar_device = f"cuda:{args.ar_gpu}" if n_gpus > 1 else f"cuda:{args.classone_gpu}"

    if n_gpus == 1:
        print("\n[!] Only 1 GPU detected — both models will load on cuda:0 sequentially.")

    # ------------------------------------------------------------------ #
    # 1. ClassOne tokenizer + packed prompt (with special tokens)
    # ------------------------------------------------------------------ #
    print(f"\n[1/4] Loading tokenizer from: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    builder = ClassOnePromptBuilder(tokenizer)

    state = {
        "customer": "Jordan M.",
        "transaction_id": 99882,
        "message": "Payment failed but funds were deducted from my checking account.",
    }
    questions = {
        "is_fraud": NoulQuestion(instructions="Is this a potential fraud event?"),
        "queue": ChoiceQuestion(
            instructions="Routing queue:",
            criteria={
                "billing": "Billing and payment disputes",
                "support": "General customer support",
                "tech": "Technical errors and outages",
            },
        ),
        "urgency": ScoreQuestion(
            instructions="Urgency level:",
            criteria=["low", "medium", "critical"],
        ),
    }
    packed = builder.pack(state=state, questions=questions)
    seq_len = packed.input_ids.shape[1]
    print(f"     ClassOne packed prompt — {seq_len} tokens")

    # AR baseline uses the ORIGINAL Gemma tokenizer (no ClassOne special tokens)
    # and a plain JSON prompt — exactly what a developer uses without ClassOne.
    ar_tokenizer = AutoTokenizer.from_pretrained(args.model)
    ar_prompt = (
        "<start_of_turn>user\n"
        "Given this customer event:\n"
        '{"customer": "Jordan M.", "transaction_id": 99882, '
        '"message": "Payment failed but funds were deducted from my checking account."}\n\n'
        "Answer in JSON with three fields:\n"
        "  is_fraud (true/false), queue (billing|support|tech), urgency (low|medium|critical).\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
        "```json\n"
    )
    ar_enc = ar_tokenizer(ar_prompt, return_tensors="pt")
    ar_prompt_len = ar_enc.input_ids.shape[1]
    print(f"     AR plain-text prompt  — {ar_prompt_len} tokens")

    # ------------------------------------------------------------------ #
    # 2. Load both models (one per GPU)
    # ------------------------------------------------------------------ #
    print(f"\n[2/4] Loading ClassOne backbone → {co_device} ...")
    co_model = ClassOneModel.from_backbone(
        base_model_name_or_path=args.model,
        tokenizer=tokenizer,
        device=co_device,
        torch_dtype=torch_dtype,
    )
    trainer = ClassOneTrainer(model=co_model, prompt_builder=builder, device=co_device)
    trainer.load_checkpoint(args.checkpoint)
    co_model.eval()
    vram_co = torch.cuda.memory_allocated(args.classone_gpu) / 1e9
    print(f"     ClassOne loaded — {vram_co:.2f} GB allocated on {co_device}")

    print(f"\n[3/4] Loading CausalLM baseline → {ar_device} ...")
    # NOTE: We load the base model WITHOUT LoRA. LoRA was trained with ClassOne
    # special tokens; applying it to a plain-text generation prompt is meaningless.
    # This is the fairest baseline: the same Gemma 4 E2B-IT that ClassOne wraps,
    # but used in standard autoregressive JSON-generation mode.
    ar_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch_dtype,
        device_map=ar_device,  # pins the whole model to cuda:1
    )
    ar_model.eval()
    ar_gpu_idx = args.ar_gpu if n_gpus > 1 else args.classone_gpu
    vram_ar = torch.cuda.memory_allocated(ar_gpu_idx) / 1e9
    print(f"     AR model loaded   — {vram_ar:.2f} GB allocated on {ar_device}")

    # ------------------------------------------------------------------ #
    # 3. Benchmark helpers
    # ------------------------------------------------------------------ #
    pad_id = ar_tokenizer.pad_token_id or ar_tokenizer.eos_token_id
    ar_input = ar_enc.input_ids.to(ar_device)

    def sync_all():
        for i in range(n_gpus):
            torch.cuda.synchronize(i)

    print("\n[4/4] Running benchmarks...\n")

    # ---- Phase A: ClassOne single-pass -------------------------------- #
    print("[*] Warming up ClassOne...")
    for _ in range(args.warmup):
        with torch.no_grad():
            co_model.evaluate_packed(packed)
        torch.cuda.synchronize(args.classone_gpu)

    print("[*] Timing ClassOne single-pass forward...")
    co_latencies = []
    for _ in range(args.iterations):
        torch.cuda.synchronize(args.classone_gpu)
        t0 = time.perf_counter()
        with torch.no_grad():
            co_model.evaluate_packed(packed)
        torch.cuda.synchronize(args.classone_gpu)
        co_latencies.append((time.perf_counter() - t0) * 1000.0)

    co_mean = float(np.mean(co_latencies))
    co_p50 = float(np.percentile(co_latencies, 50))
    co_p95 = float(np.percentile(co_latencies, 95))
    co_p99 = float(np.percentile(co_latencies, 99))
    co_rps = 1000.0 / co_mean
    print(f"    → mean {co_mean:.2f} ms  |  {co_rps:.0f} req/s")

    # ---- Phase B: Autoregressive 50-token ----------------------------- #
    print(f"\n[*] Warming up AR baseline on {ar_device}...")
    for _ in range(args.warmup):
        with torch.no_grad():
            ar_model.generate(ar_input, max_new_tokens=50, do_sample=False, pad_token_id=pad_id)
        torch.cuda.synchronize(ar_gpu_idx)

    print("[*] Timing autoregressive 50-token generation...")
    ar_latencies = []
    for _ in range(args.iterations):
        torch.cuda.synchronize(ar_gpu_idx)
        t0 = time.perf_counter()
        with torch.no_grad():
            ar_model.generate(ar_input, max_new_tokens=50, do_sample=False, pad_token_id=pad_id)
        torch.cuda.synchronize(ar_gpu_idx)
        ar_latencies.append((time.perf_counter() - t0) * 1000.0)

    ar_mean = float(np.mean(ar_latencies))
    ar_p50 = float(np.percentile(ar_latencies, 50))
    ar_p95 = float(np.percentile(ar_latencies, 95))
    ar_rps = 1000.0 / ar_mean
    speedup = ar_mean / co_mean
    print(f"    → mean {ar_mean:.2f} ms  |  {ar_rps:.0f} req/s")

    # ------------------------------------------------------------------ #
    # 4. Print summary table
    # ------------------------------------------------------------------ #
    W = 30
    sep = f"{'─' * W}─┼─{'─' * 20}─┼─{'─' * 20}"
    print()
    print("=" * 75)
    print("                 BENCHMARK RESULTS SUMMARY")
    print("=" * 75)
    print(f"{'Metric':<{W}} │ {'ClassOne (Single-Pass)':>20} │ {'Autoregressive (50t)':>20}")
    print(sep)
    print(f"{'Mean Latency':<{W}} │ {ms(co_mean):>20} │ {ms(ar_mean):>20}")
    print(f"{'P50 (Median)':<{W}} │ {ms(co_p50):>20} │ {ms(ar_p50):>20}")
    print(f"{'P95 Latency':<{W}} │ {ms(co_p95):>20} │ {ms(ar_p95):>20}")
    print(f"{'P99 Latency':<{W}} │ {ms(co_p99):>20} │ {'—':>20}")
    print(f"{'Throughput (req/s)':<{W}} │ {co_rps:>17.1f} r/s │ {ar_rps:>17.1f} r/s")
    print(f"{'Output Tokens':<{W}} │ {'0 tokens':>20} │ {'50 tokens':>20}")
    print(sep)
    print(f"  🚀  Speedup: ClassOne is  {speedup:.1f}×  faster than autoregressive decoding")
    print("=" * 75)

    results = {
        "hardware": {
            "gpus": [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "vram_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1),
                }
                for i in range(n_gpus)
            ],
            "classone_gpu": co_device,
            "ar_gpu": ar_device,
        },
        "model": args.model,
        "checkpoint": args.checkpoint,
        "dtype": args.dtype,
        "classone_prompt_tokens": seq_len,
        "ar_prompt_tokens": ar_prompt_len,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "classone": {
            "mean_ms": round(co_mean, 3),
            "p50_ms": round(co_p50, 3),
            "p95_ms": round(co_p95, 3),
            "p99_ms": round(co_p99, 3),
            "rps": round(co_rps, 1),
        },
        "autoregressive_50t": {
            "mean_ms": round(ar_mean, 3),
            "p50_ms": round(ar_p50, 3),
            "p95_ms": round(ar_p95, 3),
            "rps": round(ar_rps, 1),
        },
        "speedup_x": round(speedup, 2),
    }

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[✓] Results written to: {args.output_json}")

    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
