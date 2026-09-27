#!/usr/bin/env python3
"""Latency and throughput benchmark comparing ClassOne System One single-pass decisions

versus traditional autoregressive text generation.
"""

import argparse
import time

import numpy as np
import torch

from classone.modeling.configuration_classone import ClassOneConfig
from classone.modeling.modeling_classone import ClassOneModel
from classone.schemas import ChoiceQuestion, NoulQuestion, ScoreQuestion
from classone.tokenizer import ClassOnePromptBuilder


def parse_args():
    parser = argparse.ArgumentParser(description="ClassOne vs Autoregressive Latency Benchmark")
    parser.add_argument("--iterations", type=int, default=30, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help="Device to benchmark on",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine hardware device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print("===========================================================")
    print("   ClassOne System One vs Autoregressive LLM Benchmark")
    print(f"   Hardware Device: {device.upper()}")
    print(f"   Iterations: {args.iterations} (Warmup: {args.warmup})")
    print("===========================================================\n")

    # Set up standalone model and tokenizer for controlled, reproducible measurement
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocab = {"[UNK]": 0, "[PAD]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
    for i in range(100):
        vocab[f"w_{i}"] = 5 + i
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]")
    builder = ClassOnePromptBuilder(tokenizer)

    config = ClassOneConfig(hidden_size=256, head_hidden_size=128)
    model = ClassOneModel(config).to(device)
    model.eval()

    state = {
        "customer": "Jordan M.",
        "transaction_id": 99882,
        "message": "Payment failed but funds were deducted from checking account.",
    }
    questions = {
        "is_fraud": NoulQuestion(instructions="Is this a potential fraud event?"),
        "queue": ChoiceQuestion(
            instructions="Routing queue:",
            criteria={"billing": "Billing", "support": "General Support", "tech": "Tech Support"},
        ),
        "urgency": ScoreQuestion(
            instructions="Urgency level:",
            criteria=["low", "medium", "critical"],
        ),
    }

    packed = builder.pack(state=state, questions=questions)
    input_ids = packed.input_ids.to(device)

    # 1. Benchmark ClassOne System One (Single-Pass Forward)
    print("[*] Running ClassOne System One Single-Pass Benchmark...")
    classone_latencies = []

    # Warmup
    for _ in range(args.warmup):
        with torch.no_grad():
            _ = model.evaluate_packed(packed)
            if device == "cuda":
                torch.cuda.synchronize()
            elif device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()

    for _ in range(args.iterations):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.evaluate_packed(packed)
            if device == "cuda":
                torch.cuda.synchronize()
            elif device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
        t1 = time.perf_counter()
        classone_latencies.append((t1 - t0) * 1000.0)  # ms

    # 2. Benchmark Simulated Autoregressive Generation (50 decode steps)
    print("[*] Running Autoregressive Generative Benchmark (50 decode tokens)...")
    ar_latencies = []

    # Warmup
    for _ in range(args.warmup):
        with torch.no_grad():
            cur_ids = input_ids.clone()
            for _ in range(50):
                _ = model.extract_hidden_states(cur_ids)
                next_tok = torch.tensor([[1]], device=device)
                cur_ids = torch.cat([cur_ids, next_tok], dim=1)
            if device == "cuda":
                torch.cuda.synchronize()
            elif device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()

    for _ in range(args.iterations):
        t0 = time.perf_counter()
        with torch.no_grad():
            cur_ids = input_ids.clone()
            for _ in range(50):
                _ = model.extract_hidden_states(cur_ids)
                next_tok = torch.tensor([[1]], device=device)
                cur_ids = torch.cat([cur_ids, next_tok], dim=1)
            if device == "cuda":
                torch.cuda.synchronize()
            elif device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
        t1 = time.perf_counter()
        ar_latencies.append((t1 - t0) * 1000.0)  # ms

    # Compute Statistics
    classone_mean = np.mean(classone_latencies)
    classone_p50 = np.percentile(classone_latencies, 50)
    classone_p95 = np.percentile(classone_latencies, 95)
    classone_p99 = np.percentile(classone_latencies, 99)
    classone_rps = 1000.0 / classone_mean

    ar_mean = np.mean(ar_latencies)
    ar_p50 = np.percentile(ar_latencies, 50)
    speedup = ar_mean / classone_mean

    print("\n===========================================================")
    print("                 BENCHMARK RESULTS SUMMARY                 ")
    print("===========================================================")
    print(f"{'Metric':<28} | {'ClassOne (System One)':<18} | {'Autoregressive (50t)':<18}")
    print(f"{'-' * 28}-+-{'-' * 18}-+-{'-' * 18}")
    print(f"{'Mean Latency':<28} | {classone_mean:>14.2f} ms | {ar_mean:>14.2f} ms")
    print(f"{'Median (P50) Latency':<28} | {classone_p50:>14.2f} ms | {ar_p50:>14.2f} ms")
    print(f"{'P95 Latency':<28} | {classone_p95:>14.2f} ms | {'N/A':>17}")
    print(f"{'P99 Latency':<28} | {classone_p99:>14.2f} ms | {'N/A':>17}")
    print(f"{'Throughput (req/sec)':<28} | {classone_rps:>14.1f} req/s | {1000.0 / ar_mean:>14.1f} req/s")
    print(f"{'Output Token Cost':<28} | {'$0 (0 decode)':>18} | {'$ (50 tokens)':>18}")
    print("===========================================================")
    print(f"🚀 Speedup Factor: ClassOne is {speedup:.1f}x FASTER than generative autoregression.")
    print("===========================================================\n")


if __name__ == "__main__":
    main()
