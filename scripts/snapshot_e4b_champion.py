#!/usr/bin/env python3
"""Archives the milestone Gemma 4 E4B Champion checkpoint to snapshots directory."""

import json
import os
import shutil
import time

SNAPSHOT_NAME = "best_e4b_70pct_milestone"
SOURCE_CHECKPOINT_DIR = "worktrees/gemma-4-e4b/checkpoints/classone_e4b_champion"
SNAPSHOTS_BASE_DIR = "worktrees/gemma-4-e4b/checkpoints/snapshots"
DEST_SNAPSHOT_DIR = os.path.join(SNAPSHOTS_BASE_DIR, SNAPSHOT_NAME)

METADATA = {
    "snapshot_name": SNAPSHOT_NAME,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "base_model": "google/gemma-4-E4B-it",
    "architecture": "Gemma4ForConditionalGeneration (42 layers, d_model=2560, intermediate=10240)",
    "training_corpus": "data/champion_70pct_final_corpus.jsonl",
    "total_training_samples": 23503,
    "hardware": "2x NVIDIA GeForce RTX 5060 Ti (16 GB each, Windows Shared-Memory IPC)",
    "quantization": "4-bit NF4 base weights with BF16 compute",
    "lora_parameters": 34881536,
    "benchmarks": {
        "jevbench": {
            "overall_accuracy": "70.1% (162/231 correct) [MILESTONE REACHED]",
            "easy_tier": "100.0% (48/48 correct, 0.0000 ECE, 0.0000 Brier, 98.5ms latency)",
            "original_tier": "93.1% (67/72 correct, 100% Noul, 100% Score, 86.1% Choice, 0.0588 ECE)",
            "hard_tier": "42.3% (47/111 correct, 43.3% Choice, 42.1% Noul, 33.3% Score)",
        },
        "rlcd_alignbench": {
            "balanced_accuracy": "63.3% (63.0% standard accuracy) [MILESTONE REACHED]",
            "honesty_accuracy": "72.7% (0.567 AUROC, 0.2531 ECE)",
            "refusal_accuracy": "72.7% (0.433 AUROC)",
            "faithfulness_accuracy": "66.7% (0.600 AUROC)",
            "reward_hacking_accuracy": "66.7% (0.650 AUROC)",
            "power_seeking_accuracy": "66.7% (0.556 AUROC)",
            "privacy_accuracy": "64.3% (0.571 AUROC)",
            "uncertainty_accuracy": "64.3% (0.510 AUROC)",
        },
    },
}


def main():
    print(f"[*] Archiving {SNAPSHOT_NAME}...")
    os.makedirs(DEST_SNAPSHOT_DIR, exist_ok=True)

    # 1. Copy calibrated heads
    src_heads = os.path.join(SOURCE_CHECKPOINT_DIR, "classone_heads.pt")
    dst_heads = os.path.join(DEST_SNAPSHOT_DIR, "classone_heads.pt")
    if os.path.exists(src_heads):
        shutil.copy2(src_heads, dst_heads)
        print(f"[✓] Copied classone_heads.pt to {dst_heads}")

    # 2. Write metadata
    meta_path = os.path.join(DEST_SNAPSHOT_DIR, "snapshot_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(METADATA, f, indent=2)
    print(f"[✓] Serialized metadata to {meta_path}")

    # Also archive in root checkpoints/snapshots
    root_snap = os.path.join("checkpoints", "snapshots", SNAPSHOT_NAME)
    os.makedirs(root_snap, exist_ok=True)
    if os.path.exists(src_heads):
        shutil.copy2(src_heads, os.path.join(root_snap, "classone_heads.pt"))
    with open(os.path.join(root_snap, "snapshot_metadata.json"), "w") as f:
        json.dump(METADATA, f, indent=2)
    print(f"[✓] Synced snapshot to root: {root_snap}")


if __name__ == "__main__":
    main()
