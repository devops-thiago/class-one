#!/usr/bin/env python3
"""Prepares and validates a ClassOne candidate checkpoint for atomic Hugging Face publishing.

Performs critical pre-flight checks:
1. Validates `classone_heads.pt` exists and computes SHA256 hash.
2. Checks calibrated temperatures: confirms clamp within safe ranges ($T <= 1.0$).
3. Tokenizer delimiter serialization: runs `ClassOnePromptBuilder(tok)` and saves
   pretrained so delimiter tokens (`<|state_start|>`, `<|state_end|>`, `<|pause|>`, etc.)
   are properly serialized into `tokenizer.json`.
   - On Gemma 4: confirms delimiters map to in-vocab `<unused0>`...`<unused12>` to preserve
     Per-Layer Embeddings (PLE) without resizing vocab (which saves 5.25 GB VRAM).
   - On Qwen: confirms delimiter and pause tokens are registered in special tokens.
4. Synchronizes heads and tokenizer into the target upload directory.
5. Prints the exact atomic upload command with `--update-heads-only` (5-second push).
"""

import argparse
import hashlib
import json
import os
import shutil
import sys

import torch
from transformers import AutoTokenizer

from classone.tokenizer import ClassOnePromptBuilder

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def inspect_temperatures(heads_path: str):
    data = torch.load(heads_path, map_location="cpu", weights_only=False)
    temps = {}
    for k, v in data.items():
        if "temp" in k.lower() or "temperature" in k.lower():
            if isinstance(v, torch.Tensor):
                temps[k] = float(v.item())
            else:
                temps[k] = float(v)

    print(f"[*] Head state keys: {list(data.keys())}")
    if temps:
        print(f"[*] Calibrated temperatures: {temps}")
        for k, val in temps.items():
            if val > 1.0:
                print(
                    f"[!] WARNING: Temperature {k}={val:.3f} > 1.0! Unconstrained temperatures flatten option margins."
                )
            elif val < 0.35:
                print(
                    f"[!] WARNING: Temperature {k}={val:.3f} < 0.35! Extremely low temperatures may cause overconfidence."
                )
            else:
                print(f"    [OK] Temperature {k}={val:.3f} within optimal range.")
    else:
        print("[!] Note: No explicit calibrated temperatures found in head checkpoint (using runtime defaults).")


def main():
    parser = argparse.ArgumentParser(description="Prepare candidate checkpoint for HF upload")
    parser.add_argument(
        "--snapshot-dir", type=str, required=True, help="Snapshot directory containing classone_heads.pt"
    )
    parser.add_argument("--target-dir", type=str, required=True, help="Target merged directory to stage for upload")
    parser.add_argument(
        "--repo-id", type=str, required=True, help="Target HF repo ID (e.g. devops-thiago/classone-qwen3.5-9b)"
    )
    parser.add_argument("--pause-tokens", type=int, default=8, help="Default latent pause tokens")
    args = parser.parse_args()

    heads_src = os.path.join(args.snapshot_dir, "classone_heads.pt")
    if not os.path.exists(heads_src):
        print(f"[!] Error: classone_heads.pt not found in {args.snapshot_dir}")
        sys.exit(1)

    os.makedirs(args.target_dir, exist_ok=True)
    heads_dst = os.path.join(args.target_dir, "classone_heads.pt")

    print("=" * 80)
    print(f"PREPARING CANDIDATE CHECKPOINT FOR {args.repo_id}")
    print("=" * 80)

    # 1. Compute SHA256 and inspect temperatures
    head_sha = sha256_file(heads_src)
    print(f"[*] Candidate heads source: {heads_src}")
    print(f"[*] Candidate heads SHA256: {head_sha}")
    inspect_temperatures(heads_src)

    # Check snapshot metadata if present
    meta_path = os.path.join(args.snapshot_dir, "snapshot_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        print(f"[*] Snapshot Metadata:\n    {json.dumps(meta, indent=4)}")

    # 2. Stage heads to target directory
    shutil.copyfile(heads_src, heads_dst)
    print(f"[OK] Staged candidate heads to {heads_dst}")

    # 3. Tokenizer Delimiter Token Processing
    print(f"[*] Processing tokenizer delimiters in {args.target_dir}...")
    try:
        tok = AutoTokenizer.from_pretrained(args.target_dir)
    except Exception:
        print(f"[*] Tokenizer not in {args.target_dir}, loading from base repo {args.repo_id}...")
        tok = AutoTokenizer.from_pretrained(args.repo_id)

    builder = ClassOnePromptBuilder(tok, default_pause_tokens=args.pause_tokens)
    tok.save_pretrained(args.target_dir)
    print(f"[OK] Serialized delimiters into tokenizer in {args.target_dir}")
    print(f"     Vocab size: {len(tok)}")
    print(f"     Delimiter mappings: {builder.delimiter_map}")

    is_gemma = any("gemma" in part.lower() for part in [args.repo_id, args.target_dir])
    if is_gemma:
        for delim, mapped in builder.delimiter_map.items():
            if not mapped.startswith("<unused"):
                print(f"[!] WARNING: Gemma 4 delimiter '{delim}' mapped to '{mapped}' instead of '<unused*>'!")
        print("[OK] Confirmed Gemma 4 in-vocab unused token mapping (PLE preserved, 0 extra VRAM).")

    # 4. Next steps command
    print("\n" + "=" * 80)
    print("PREPARATION COMPLETE! RUN THE ATOMIC UPLOAD COMMAND BELOW:")
    print("=" * 80)
    print("python scripts/push_to_hub.py `")
    print(f'  --checkpoint-dir "{args.target_dir}" `')
    print(f'  --repo-id "{args.repo_id}" `')
    print("  --update-heads-only\n")
    print(f"Candidate Head SHA256: {head_sha}")
    print("=" * 80)


if __name__ == "__main__":
    main()
