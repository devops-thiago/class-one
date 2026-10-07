#!/usr/bin/env python3
"""Archives and snapshots the best model states for rollback safety."""

import json
import os
import shutil
import time

SNAPSHOT_BASE = "checkpoints/snapshots"


def snapshot_model(name: str, checkpoint_dir: str, merged_dir: str, metadata: dict):
    target_dir = os.path.join(SNAPSHOT_BASE, name)
    os.makedirs(target_dir, exist_ok=True)

    # 1. Copy heads
    src_heads = os.path.join(checkpoint_dir, "classone_heads.pt")
    if os.path.exists(src_heads):
        shutil.copy2(src_heads, os.path.join(target_dir, "classone_heads.pt"))
        print(f"[✓] Saved heads to {target_dir}/classone_heads.pt")

    # 2. Copy adapter config if exists
    src_adapter_cfg = os.path.join(checkpoint_dir, "lora_backbone", "adapter_config.json")
    if os.path.exists(src_adapter_cfg):
        os.makedirs(os.path.join(target_dir, "lora_backbone"), exist_ok=True)
        shutil.copy2(src_adapter_cfg, os.path.join(target_dir, "lora_backbone", "adapter_config.json"))

    # 3. Save metadata
    metadata["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    metadata["checkpoint_dir"] = checkpoint_dir
    metadata["merged_dir"] = merged_dir
    with open(os.path.join(target_dir, "snapshot_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[✓] Snapshot '{name}' saved successfully in {target_dir}")


def main():
    # Snapshot A: Best Continual Learning Model
    snapshot_model(
        name="best_continual_59pct",
        checkpoint_dir="checkpoints/classone_gemma4_e2b",
        merged_dir="checkpoints/classone_gemma4_e2b_merged",
        metadata={
            "description": "Best continual learning model on 12,004 dataset with preserved heads",
            "jevbench_overall": 59.3,
            "jevbench_easy": 95.8,
            "jevbench_original": 68.1,
            "jevbench_hard": 40.5,
            "alignbench_balanced_acc": 53.0,
            "alignbench_honesty_auroc": 0.733,
            "alignbench_power_seeking_auroc": 0.667,
            "alignbench_refusal_auroc": 0.733,
        },
    )

    # Snapshot B: Best Clean Scratch Model
    snapshot_model(
        name="best_scratch_42pct_hard",
        checkpoint_dir="checkpoints/classone_clean_scratch",
        merged_dir="checkpoints/classone_clean_scratch_merged",
        metadata={
            "description": "Clean single-stage model from google/gemma-4-e2b-it on 12,004 dataset",
            "jevbench_overall": 48.5,
            "jevbench_hard_choice": 41.8,
            "alignbench_reward_hacking_auroc": 0.600,
            "alignbench_power_seeking_auroc": 0.778,
            "alignbench_sycophancy_auroc": 0.450,
        },
    )


if __name__ == "__main__":
    main()
