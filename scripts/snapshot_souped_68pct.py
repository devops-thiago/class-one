#!/usr/bin/env python3
"""Archives the 68.0% record souped model."""

from create_snapshot import snapshot_model


def main():
    snapshot_model(
        name="best_souped_68pct",
        checkpoint_dir="checkpoints/classone_70pct_milestone",
        merged_dir="checkpoints/classone_70pct_milestone_merged",
        metadata={
            "description": "All-time record model: 68.0% on JevBench (157/231 correct) with Model Souping (alpha=0.50)",
            "jevbench_overall": 68.0,
            "jevbench_easy": 97.9,
            "jevbench_easy_choice": 100.0,
            "jevbench_easy_noul": 91.7,
            "jevbench_original": 83.3,
            "jevbench_original_choice": 80.6,
            "jevbench_original_score": 100.0,
            "jevbench_original_noul": 79.2,
            "jevbench_hard": 45.0,
            "jevbench_hard_choice": 43.3,
            "jevbench_hard_noul": 55.3,
        },
    )


if __name__ == "__main__":
    main()
