#!/usr/bin/env python3
"""Archives the 2-epoch cosine annealed model."""

from create_snapshot import snapshot_model


def main():
    snapshot_model(
        name="best_scratch_cosine_2ep",
        checkpoint_dir="checkpoints/classone_scratch_cosine",
        merged_dir="checkpoints/classone_scratch_cosine_merged",
        metadata={
            "description": "2-epoch cosine annealed model from google/gemma-4-e2b-it with warmup and lr decay",
            "jevbench_overall": 59.7,
            "jevbench_easy": 91.7,
            "jevbench_original": 63.9,
            "jevbench_original_score": 100.0,
            "jevbench_hard": 43.2,
            "jevbench_hard_choice": 41.8,
            "jevbench_hard_noul": 50.0,
            "alignbench_power_seeking_auroc": 0.889,
            "alignbench_honesty_auroc": 0.767,
            "alignbench_uncertainty_auroc": 0.755,
            "alignbench_refusal_auroc": 0.667,
        },
    )


if __name__ == "__main__":
    main()
