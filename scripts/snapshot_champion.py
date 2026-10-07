#!/usr/bin/env python3
"""Archives the Champion model."""

from create_snapshot import snapshot_model


def main():
    snapshot_model(
        name="best_champion_70pct",
        checkpoint_dir="checkpoints/classone_champion_70pct",
        merged_dir="checkpoints/classone_champion_70pct_merged",
        metadata={
            "description": "Champion 23.5k dataset model: 84.7% on Original (87.5% Noul), 95.8% Easy, 54.1% AlignBench Balanced Acc (0.889 Power Seeking, 0.0460 Honesty ECE)",
            "jevbench_overall": 66.2,
            "jevbench_easy": 95.8,
            "jevbench_original": 84.7,
            "jevbench_hard": 41.4,
            "alignbench_balanced_acc": 54.1,
            "alignbench_power_seeking_auroc": 0.889,
            "alignbench_honesty_ece": 0.0460,
            "alignbench_privacy_acc": 42.9,
            "alignbench_reward_hacking_acc": 44.4,
        },
    )


if __name__ == "__main__":
    main()
