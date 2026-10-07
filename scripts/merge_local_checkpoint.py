#!/usr/bin/env python3
"""Merges fine-tuned LoRA adapters into the base model weights."""

import os

import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

CHECKPOINT_DIR = "checkpoints/classone_gemma4_e2b"
LORA_DIR = os.path.join(CHECKPOINT_DIR, "lora_backbone")
BASE_MODEL = "devops-thiago/classone-gemma4-e2b"
OUTPUT_DIR = "checkpoints/classone_gemma4_e2b_merged"


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=BASE_MODEL, help="Base model identifier or path")
    parser.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR, help="Directory containing lora_backbone")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Destination directory for merged weights")
    args = parser.parse_args()

    lora_dir = os.path.join(args.checkpoint_dir, "lora_backbone")
    if not os.path.exists(lora_dir):
        print(f"[!] LoRA directory not found: {lora_dir}")
        return

    print(f"[*] Loading base model {args.base}...")
    base = AutoModel.from_pretrained(
        args.base,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )

    print(f"[*] Loading LoRA adapter from {lora_dir}...")
    peft_model = PeftModel.from_pretrained(base, lora_dir)

    print("[*] Merging adapter into base weights...")
    merged = peft_model.merge_and_unload()

    print(f"[*] Saving merged weights to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    merged.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size="5GB")

    # Copy tokenizer with ClassOne special tokens
    from classone.tokenizer import ClassOnePromptBuilder

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    ClassOnePromptBuilder(tokenizer)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[✓] Merged model and tokenizer saved to {args.output_dir}")


if __name__ == "__main__":
    main()
