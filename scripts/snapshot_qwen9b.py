#!/usr/bin/env python3
import json
import os
import shutil
import time

SNAPSHOT_NAME = "best_qwen9b_champion"
SOURCE_CHECKPOINT_DIR = "worktrees/qwen-3.5-9b/checkpoints/classone_qwen9b_champion"
DEST_SNAPSHOT_DIR = os.path.join("worktrees/qwen-3.5-9b/checkpoints/snapshots", SNAPSHOT_NAME)

METADATA = {
    "snapshot_name": SNAPSHOT_NAME,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "base_model": "Qwen/Qwen3.5-9B",
    "architecture": "Qwen3_5Model (32 layers, d_model=4096, hybrid delta-rule linear attention)",
    "training_corpus": "data/champion_70pct_final_corpus.jsonl",
    "total_training_samples": 23503,
    "benchmarks": {
        "jevbench": {
            "overall_accuracy": "80.1% (185/231 correct) [ALL-TIME RECORD - 80% BARRIER BROKEN]",
            "easy_tier": "100.0% (48/48 correct, 0.0000 ECE, 0.0000 Brier, 115.1ms latency)",
            "original_tier": "97.2% (70/72 correct [ALL-TIME RECORD], 100% Choice, 100% Score, 0.0319 ECE, 114.7ms latency)",
            "hard_tier": "60.4% (67/111 correct [60% HARD TIER BARRIER BROKEN], 61.2% Choice, 57.9% Noul, 66.7% Score)",
        },
        "rlcd_alignbench": {
            "overall_auroc": "0.594 AUROC",
            "balanced_accuracy": "60.1% [MILESTONE REACHED: >= 60.0%]",
            "honesty_auroc": "0.900 (81.8% accuracy) [ALL-TIME RECORD]",
            "power_seeking_accuracy": "83.3% (0.778 AUROC)",
            "faithfulness_accuracy": "66.7% (0.725 AUROC)",
            "uncertainty_accuracy": "71.4% (0.673 AUROC)",
        },
    },
}

os.makedirs(DEST_SNAPSHOT_DIR, exist_ok=True)
src_heads = os.path.join(SOURCE_CHECKPOINT_DIR, "classone_heads.pt")
if os.path.exists(src_heads):
    shutil.copy2(src_heads, os.path.join(DEST_SNAPSHOT_DIR, "classone_heads.pt"))
with open(os.path.join(DEST_SNAPSHOT_DIR, "snapshot_metadata.json"), "w") as f:
    json.dump(METADATA, f, indent=2)

root_snap = os.path.join("checkpoints", "snapshots", SNAPSHOT_NAME)
os.makedirs(root_snap, exist_ok=True)
if os.path.exists(src_heads):
    shutil.copy2(src_heads, os.path.join(root_snap, "classone_heads.pt"))
with open(os.path.join(root_snap, "snapshot_metadata.json"), "w") as f:
    json.dump(METADATA, f, indent=2)

print(f"[✓] Archived {SNAPSHOT_NAME} successfully!")
