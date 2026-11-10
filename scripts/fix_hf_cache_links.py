#!/usr/bin/env python3
"""Fixes Windows Hugging Face cache snapshot symlink limitation using NTFS hardlinks.

Problem:
On Windows without Developer Mode, Hugging Face Hub cannot create symlinks across
snapshots in `E:\\huggingface\\hub` (or `~/.cache/huggingface/hub`).
When a new commit updates only small files (e.g. `classone_heads.pt`, `README.md`),
HF Hub creates a new snapshot folder that lacks the 18-35 GB `.safetensors` files.
When `from_pretrained` runs against the new commit, it triggers a redundant full
download of 18-35 GB over the network.

Solution:
This script scans snapshot directories for a given model (or all ClassOne models),
finds any snapshot missing `.safetensors` files, and creates NTFS hardlinks (`os.link`)
from existing snapshots or the `blobs/` store.
Result: 0 bytes downloaded, 0 bytes extra disk space, instant execution (~0.01s).
"""

import argparse
import os
import shutil
import sys

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


DEFAULT_CACHE_DIRS = [
    os.environ.get("HUGGINGFACE_HUB_CACHE"),
    os.environ.get("HF_HOME", "") + "/hub" if os.environ.get("HF_HOME") else None,
    "E:\\huggingface\\hub",
    "E:/huggingface/hub",
    os.path.expanduser("~/.cache/huggingface/hub"),
]


def find_hf_cache() -> str | None:
    for p in DEFAULT_CACHE_DIRS:
        if p and os.path.exists(p):
            return os.path.abspath(p)
    return None


def fix_model_snapshots(model_dir: str, dry_run: bool = False) -> int:
    snapshots_dir = os.path.join(model_dir, "snapshots")
    if not os.path.exists(snapshots_dir):
        return 0

    snapshot_hashes = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
    if len(snapshot_hashes) <= 1:
        return 0

    # Collect all known weight/safetensors files across all snapshots
    known_weights = {}  # filename -> existing_source_path
    for sh in snapshot_hashes:
        s_path = os.path.join(snapshots_dir, sh)
        for f in os.listdir(s_path):
            if f.endswith(".safetensors") or f.endswith(".bin") or f.endswith(".index.json"):
                full_f = os.path.join(s_path, f)
                if f not in known_weights and os.path.isfile(full_f):
                    known_weights[f] = full_f

    if not known_weights:
        return 0

    linked_count = 0
    model_name = os.path.basename(model_dir).replace("models--", "").replace("--", "/")
    print(f"\n[*] Checking cache snapshots for: {model_name}")
    print(f"    Available weight files: {list(known_weights.keys())}")

    for sh in snapshot_hashes:
        s_path = os.path.join(snapshots_dir, sh)
        existing_files = set(os.listdir(s_path))

        for w_name, src_path in known_weights.items():
            if w_name not in existing_files:
                dst_path = os.path.join(s_path, w_name)
                sz_gb = os.path.getsize(src_path) / (1024**3)
                print(f"    [+] Snapshot {sh[:10]}... missing {w_name} ({sz_gb:.2f} GB)")

                if not dry_run:
                    try:
                        # Attempt NTFS hardlink
                        os.link(src_path, dst_path)
                        print(f"        -> Hardlinked successfully: {w_name} (0 bytes disk cost)")
                        linked_count += 1
                    except OSError as exc:
                        print(f"        -> Hardlink failed ({exc}), falling back to copy...")
                        shutil.copyfile(src_path, dst_path)
                        print(f"        -> Copied successfully: {w_name}")
                        linked_count += 1
                else:
                    print(f"        -> [DRY-RUN] Would hardlink {src_path} -> {dst_path}")
                    linked_count += 1

    return linked_count


def main():
    parser = argparse.ArgumentParser(description="Fix Windows HF Cache snapshots missing safetensors")
    parser.add_argument(
        "--repo-id", type=str, default=None, help="Specific model repo (e.g. devops-thiago/classone-qwen3.5-9b)"
    )
    parser.add_argument("--cache-dir", type=str, default=None, help="Custom HF cache hub directory path")
    parser.add_argument("--dry-run", action="store_true", help="Report missing files without modifying")
    args = parser.parse_args()

    cache_dir = args.cache_dir or find_hf_cache()
    if not cache_dir or not os.path.exists(cache_dir):
        print(f"[!] Error: Could not locate Hugging Face Hub cache directory. Searched: {DEFAULT_CACHE_DIRS}")
        sys.exit(1)

    print(f"[*] Using Hugging Face Hub cache at: {cache_dir}")

    total_linked = 0
    if args.repo_id:
        folder_name = "models--" + args.repo_id.replace("/", "--")
        target_dir = os.path.join(cache_dir, folder_name)
        if not os.path.exists(target_dir):
            print(f"[!] Model folder not found in cache: {target_dir}")
            sys.exit(1)
        total_linked += fix_model_snapshots(target_dir, dry_run=args.dry_run)
    else:
        # Check all models matching classone
        for d in os.listdir(cache_dir):
            if d.startswith("models--") and ("classone" in d.lower() or "qwen" in d.lower() or "gemma" in d.lower()):
                full_d = os.path.join(cache_dir, d)
                if os.path.isdir(full_d):
                    total_linked += fix_model_snapshots(full_d, dry_run=args.dry_run)

    print(f"\n[✓] Finished cache scan. Total files linked/fixed: {total_linked}")


if __name__ == "__main__":
    main()
