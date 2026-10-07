#!/usr/bin/env python3
"""Archives the Original-record 86.1% model."""

from create_snapshot import snapshot_model


def main():
    snapshot_model(
        name="best_sota_original_86pct",
        checkpoint_dir="checkpoints/classone_sota_70pct_final",
        merged_dir="checkpoints/classone_sota_70pct_final_merged",
        metadata={
            "description": "Original Tier record model: 86.1% on Original (62/72 correct, 100% Score, 87.5% Noul, 0.0748 ECE), 97.9% on Easy (0.0056 ECE)",
            "jevbench_overall": 67.5,
            "jevbench_easy": 97.9,
            "jevbench_easy_choice": 100.0,
            "jevbench_easy_noul": 91.7,
            "jevbench_easy_ece": 0.0056,
            "jevbench_original": 86.1,
            "jevbench_original_choice": 80.6,
            "jevbench_original_score": 100.0,
            "jevbench_original_noul": 87.5,
            "jevbench_original_ece": 0.0748,
            "jevbench_hard": 42.3,
            "alignbench_power_seeking_auroc": 0.889,
            "alignbench_honesty_acc": 72.7,
        },
    )


if __name__ == "__main__":
    main()
