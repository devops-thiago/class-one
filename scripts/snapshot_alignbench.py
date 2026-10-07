#!/usr/bin/env python3
"""Archives the AlignBench record 56.1% model."""

from create_snapshot import snapshot_model


def main():
    snapshot_model(
        name="best_alignbench_56pct",
        checkpoint_dir="checkpoints/classone_70pct_final_consolidated",
        merged_dir="checkpoints/classone_70pct_final_consolidated_merged",
        metadata={
            "description": "AlignBench record model: 56.1% balanced accuracy (83.3% Power Seeking, 72.7% Honesty, 71.4% Uncertainty, 0.700 Faithfulness AUROC)",
            "alignbench_balanced_acc": 56.1,
            "alignbench_power_seeking_acc": 83.3,
            "alignbench_power_seeking_auroc": 0.889,
            "alignbench_honesty_acc": 72.7,
            "alignbench_uncertainty_acc": 71.4,
            "alignbench_faithfulness_auroc": 0.700,
            "alignbench_prompt_injection_acc": 62.5,
            "alignbench_refusal_acc": 63.6,
            "jevbench_overall": 64.9,
            "jevbench_easy": 97.9,
            "jevbench_original": 81.9,
        },
    )


if __name__ == "__main__":
    main()
