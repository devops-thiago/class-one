#!/usr/bin/env python3
"""Archives the ContractNLI-boosted 62.8% record model."""

from create_snapshot import snapshot_model


def main():
    snapshot_model(
        name="best_contract_boost_63pct",
        checkpoint_dir="checkpoints/classone_hard_contract_boost",
        merged_dir="checkpoints/classone_hard_contract_boost_merged",
        metadata={
            "description": "All-time record model: 62.8% on JevBench (145/231 correct) with ContractNLI & CUAD",
            "jevbench_overall": 62.8,
            "jevbench_easy": 95.8,
            "jevbench_easy_choice": 100.0,
            "jevbench_easy_noul": 83.3,
            "jevbench_original": 69.4,
            "jevbench_original_choice": 72.2,
            "jevbench_original_score": 100.0,
            "jevbench_hard": 44.1,
            "jevbench_hard_choice": 43.3,
            "jevbench_hard_noul": 50.0,
            "alignbench_balanced_acc": 53.2,
            "alignbench_faithfulness_auroc": 0.650,
            "alignbench_uncertainty_auroc": 0.735,
            "alignbench_power_seeking_auroc": 0.778,
        },
    )


if __name__ == "__main__":
    main()
