#!/usr/bin/env python3
import json
import os
import shutil
import time

SNAPSHOT_NAME = "best_qwen2b_champion"
SOURCE_CHECKPOINT_DIR = "worktrees/qwen-3.5-2b/checkpoints/classone_qwen2b_champion"
DEST_SNAPSHOT_DIR = os.path.join("worktrees/qwen-3.5-2b/checkpoints/snapshots", SNAPSHOT_NAME)

METADATA = {
    "snapshot_name": SNAPSHOT_NAME,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "base_model": "Qwen/Qwen3.5-2B",
    "architecture": "Qwen3_5Model (24 layers, d_model=2048, hybrid delta-rule linear attention)",
    "training_corpus": "data/champion_70pct_final_corpus.jsonl",
    "total_training_samples": 23503,
    "benchmarks": {
        "jevbench": {
            "overall_accuracy": "68.0% (157/231 correct)",
            "easy_tier": "100.0% (48/48 correct, 0.0014 ECE, 41.9ms latency)",
            "original_tier": "91.7% (66/72 correct, 88.9% Choice [RECORD], 100% Score, 40.3ms latency)",
            "hard_tier": "38.7% (43/111 correct, 47.4% Noul, 34.3% Choice)",
        },
        "rlcd_alignbench": {
            "balanced_accuracy": "58.0%",
            "faithfulness_auroc": "0.800 [ALL-TIME RECORD]",
            "honesty_accuracy": "72.7% (0.767 AUROC)",
            "latency_p50_ms": "173.2 ms (5.3 decisions/sec)",
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
