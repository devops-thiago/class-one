#!/usr/bin/env python3
"""Interpolates (soups) decision heads from Enterprise Boost and 70% Milestone."""

import torch

HEADS_A_PATH = "checkpoints/classone_enterprise_boost/classone_heads.pt"
HEADS_B_PATH = "checkpoints/classone_70pct_milestone/classone_heads.pt"
OUTPUT_SOUP_PATH = "checkpoints/classone_heads_soup_70.pt"


def main():
    heads_a = torch.load(HEADS_A_PATH, map_location="cpu")
    heads_b = torch.load(HEADS_B_PATH, map_location="cpu")

    alpha = 0.50  # Equal blend of high Hard Choice + high Original Noul
    soup_heads = {}

    for head_name in ["noul_head", "choice_head", "score_head"]:
        soup_heads[head_name] = {}
        state_a = heads_a[head_name]
        state_b = heads_b[head_name]
        for key in state_a:
            if key in state_b and state_a[key].shape == state_b[key].shape:
                if state_a[key].is_floating_point():
                    soup_heads[head_name][key] = alpha * state_a[key] + (1.0 - alpha) * state_b[key]
                else:
                    soup_heads[head_name][key] = state_a[key]
            else:
                soup_heads[head_name][key] = state_a[key]

    torch.save(soup_heads, OUTPUT_SOUP_PATH)
    print(f"[✓] Successfully created souped heads at {OUTPUT_SOUP_PATH} (alpha={alpha})")


if __name__ == "__main__":
    main()
