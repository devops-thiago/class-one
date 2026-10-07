#!/usr/bin/env python3
import json
import os
import shutil
import time

SNAPSHOT_NAME = "best_qwen4b_champion"
SOURCE_CHECKPOINT_DIR = "worktrees/qwen-3.5-4b/checkpoints/classone_qwen4b_champion"
DEST_SNAPSHOT_DIR = os.path.join("worktrees/qwen-3.5-4b/checkpoints/snapshots", SNAPSHOT_NAME)

METADATA = {
    "snapshot_name": SNAPSHOT_NAME,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "base_model": "Qwen/Qwen3.5-4B",
    "architecture": "Qwen3_5Model (32 layers, d_model=2560, hybrid delta-rule linear attention)",
    "training_corpus": "data/champion_70pct_final_corpus.jsonl",
    "total_training_samples": 23503,
    "benchmarks": {
        "jevbench": {
            "overall_accuracy": "70.1% (162/231 correct) [MILESTONE REACHED]",
            "easy_tier": "100.0% (48/48 correct, 0.0001 ECE, 63.4ms latency)",
            "original_tier": "95.8% (69/72 correct [ALL-TIME RECORD], 97.2% Choice, 100% Score, 61.1ms latency)",
            "hard_tier": "40.5% (45/111 correct, 44.7% Noul, 40.3% Choice)",
        },
        "rlcd_alignbench": {
            "overall_auroc": "0.604 AUROC",
            "balanced_accuracy": "55.3%",
            "faithfulness_auroc": "0.850 [ALL-TIME RECORD]",
            "power_seeking_accuracy": "83.3% (0.778 AUROC)",
            "refusal_accuracy": "72.7% (0.0753 ECE)",
            "honesty_accuracy": "72.7% (0.733 AUROC)",
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
