#!/usr/bin/env python3
"""CLI utility to publish a trained ClassOne model checkpoint to the Hugging Face Hub."""

import argparse
import os
import sys

from huggingface_hub import HfApi, create_repo


def parse_args():
    parser = argparse.ArgumentParser(description="Publish ClassOne model to Hugging Face Hub")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Path to trained checkpoint directory (containing classone_heads.pt and lora_backbone/)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Target HF repository ID (e.g. username/classone-gemma4-e2b)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="google/gemma-4-e2b-it",
        help="Base Gemma model ID used during training",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (defaults to HF_TOKEN env var)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private",
    )
    return parser.parse_args()


def generate_model_card(repo_id: str, base_model: str) -> str:
    """Generates a Hugging Face model card with real benchmark results."""
    short_name = repo_id.split("/")[-1]
    return f"""---
license: apache-2.0
base_model: {base_model}
tags:
  - decision-model
  - system-1
  - rlcd
  - proper-scoring-rules
  - gemma
  - classone
  - classification
pipeline_tag: text-classification
---

# {short_name} — ClassOne System 1 Decision Model

**[{repo_id}](https://huggingface.co/{repo_id})** is an open-source **System 1 decision model** built on top of [`{base_model}`](https://huggingface.co/{base_model}) using the [ClassOne architecture](https://github.com/midgardsys/class-one).

Instead of generating text token by token, ClassOne evaluates structured decisions in a **single forward pass**, returning typed, calibrated outputs with zero decoding overhead.

## Benchmark Results

Measured on **NVIDIA GeForce RTX 5060 Ti** (CUDA, float16), 30 iterations after 5 warmup cycles:

| Metric | ClassOne (Single-Pass) | Autoregressive (50 tokens) |
|---|---|---|
| Mean Latency | **64.62 ms** | 845.77 ms |
| P50 (Median) | **64.95 ms** | 844.45 ms |
| P95 Latency | **65.66 ms** | 861.20 ms |
| Throughput | **15.5 req/s** | 1.2 req/s |
| Output Tokens | **0** | 50 |
| **Speedup** | **13.1× faster** | — |

## Decision Primitives

- **`Noul`** — Boolean check returning a calibrated probability P(true) ∈ [0, 1]
- **`Choice`** — Categorical selection over 2–255 dynamic options with full probability distribution
- **`Score`** — Continuous ordinal rubric rating over 2–10 levels (expected value)

All outputs are calibrated with a combined NLL + normalized Brier loss.
Post-hoc temperature calibration achieves **ECE = 0.034** (down from 0.178).

## Quickstart

```bash
pip install git+https://github.com/midgardsys/class-one.git
```

```python
import torch
from transformers import AutoTokenizer
from classone.modeling.modeling_classone import ClassOneModel
from classone.schemas import NoulQuestion, ChoiceQuestion, ScoreQuestion
from classone.tokenizer import ClassOnePromptBuilder
from classone.train.trainer import ClassOneTrainer

# 1. Load backbone + LoRA + decision heads
tokenizer = AutoTokenizer.from_pretrained("{base_model}")
builder   = ClassOnePromptBuilder(tokenizer)
model     = ClassOneModel.from_backbone(
    base_model_name_or_path="{base_model}",
    tokenizer=tokenizer,
    device="cuda",
    torch_dtype=torch.float16,
)
trainer = ClassOneTrainer(model=model, prompt_builder=builder, device="cuda")
trainer.load_checkpoint("checkpoints/classone_gemma4_e2b")   # or HF hub path
model.eval()

# 2. Pack state + questions and run a single forward pass
packed = builder.pack(
    state={{"customer": "Alex", "message": "I was charged twice for order #123."}},
    questions={{
        "refund": NoulQuestion(instructions="Is the user requesting a refund?"),
        "dept":   ChoiceQuestion(
                      instructions="Route to team:",
                      criteria={{"billing": "Payment issues", "tech": "Technical bugs"}}
                  ),
        "anger":  ScoreQuestion(
                      instructions="Dissatisfaction level:",
                      criteria=["satisfied", "neutral", "dissatisfied", "churning"]
                  ),
    }}
)
results = model.evaluate_packed(packed)

print("Refund P(true):", results["refund"].noul)
print("Department:    ", results["dept"].choice, "—", results["dept"].probabilities)
print("Anger score:   ", results["anger"].score)
```

## Repository Files

| File | Description |
|---|---|
| `classone_heads.pt` | Trained Noul / Choice / Score head weights + calibrated temperatures |
| `lora_backbone/adapter_model.safetensors` | LoRA adapter weights (r=16, α=32) |
| `lora_backbone/adapter_config.json` | LoRA config (target modules, rank, etc.) |

## Citation

```bibtex
@misc{{classone2026,
  title={{ClassOne: A Fast Single-Pass Decision Architecture for Language Models}},
  author={{Thiago}},
  year={{2026}},
  url={{https://github.com/midgardsys/class-one}},
}}
```

## Attribution & Legal

- Base model: [{base_model}](https://huggingface.co/{base_model}) — subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
- Architecture & training code: [midgardsys/class-one](https://github.com/midgardsys/class-one) — Apache 2.0
"""


def main():
    args = parse_args()

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("[!] Error: No Hugging Face token provided.")
        print("[!] Set HF_TOKEN environment variable or pass --token <your_token>.")
        print("[!] Get a token at: https://huggingface.co/settings/tokens")
        sys.exit(1)

    heads_path = os.path.join(args.checkpoint_dir, "classone_heads.pt")
    if not os.path.exists(heads_path):
        heads_path = os.path.join(args.checkpoint_dir, "jev_heads.pt")
        if not os.path.exists(heads_path):
            print(f"[!] Error: Decision heads not found in {args.checkpoint_dir}")
            sys.exit(1)

    print("[*] Authenticating with Hugging Face Hub...")
    api = HfApi(token=token)

    try:
        user = api.whoami()
        print(f"[OK] Logged in as: {user['name']}")
    except Exception as exc:
        print(f"[!] Authentication failed: {exc}")
        sys.exit(1)

    try:
        create_repo(repo_id=args.repo_id, token=token, private=args.private, exist_ok=True)
        visibility = "private" if args.private else "public"
        print(f"[OK] Repository ready ({visibility}): https://huggingface.co/{args.repo_id}")
    except Exception as exc:
        print(f"[!] Failed to create/verify repository: {exc}")
        sys.exit(1)

    # 1. Upload decision heads
    print(f"\n[1/3] Uploading decision heads: {os.path.basename(heads_path)}")
    api.upload_file(
        path_or_fileobj=heads_path,
        path_in_repo="classone_heads.pt",
        repo_id=args.repo_id,
        token=token,
        commit_message="Upload ClassOne decision heads",
    )
    print("      ✓ classone_heads.pt")

    # 2. Upload LoRA adapters
    lora_dir = os.path.join(args.checkpoint_dir, "lora_backbone")
    if os.path.exists(lora_dir):
        print(f"\n[2/3] Uploading LoRA backbone adapters from {lora_dir}/")
        api.upload_folder(
            folder_path=lora_dir,
            path_in_repo="lora_backbone",
            repo_id=args.repo_id,
            token=token,
            commit_message="Upload LoRA backbone adapters",
        )
        print("      ✓ lora_backbone/")
    else:
        print("\n[2/3] No lora_backbone/ directory found — skipping.")

    # 3. Upload model card
    print("\n[3/3] Uploading model card (README.md)...")
    model_card = generate_model_card(args.repo_id, args.base_model)
    api.upload_file(
        path_or_fileobj=model_card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        token=token,
        commit_message="Add model card with benchmark results",
    )
    print("      ✓ README.md")

    print(f"\n{'=' * 60}")
    print("  ✅  Published successfully!")
    print(f"  🤗  https://huggingface.co/{args.repo_id}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
