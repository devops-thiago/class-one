#!/usr/bin/env python3
"""Archives the Enterprise-boosted 64.5% record model."""

from create_snapshot import snapshot_model


def main():
    snapshot_model(
        name="best_enterprise_boost_65pct",
        checkpoint_dir="checkpoints/classone_enterprise_boost",
        merged_dir="checkpoints/classone_enterprise_boost_merged",
        metadata={
            "description": "All-time record model: 64.5% on JevBench (149/231 correct) with Enterprise Curriculum",
            "jevbench_overall": 64.5,
            "jevbench_easy": 97.9,
            "jevbench_easy_choice": 100.0,
            "jevbench_easy_noul": 91.7,
            "jevbench_original": 69.4,
            "jevbench_original_choice": 77.8,
            "jevbench_original_score": 91.7,
            "jevbench_hard": 46.9,
            "jevbench_hard_choice": 46.3,
            "jevbench_hard_noul": 50.0,
            "alignbench_balanced_acc": 53.3,
            "alignbench_power_seeking_auroc": 0.889,
            "alignbench_uncertainty_acc": 85.7,
            "alignbench_honesty_acc": 72.7,
            "alignbench_faithfulness_auroc": 0.600,
        },
    )


if __name__ == "__main__":
    main()
