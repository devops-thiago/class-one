#!/usr/bin/env python3
"""Soups baseline Qwen 4B heads with new FP16 Hard-Tier heads."""

import os
import shutil

import torch

HEADS_A_PATH = "checkpoints/snapshots/rollback_qwen4b_pre_fp16/classone_heads.pt"
HEADS_B_PATH = "checkpoints/qwen4b_fp16_candidate/classone_heads.pt"
OUTPUT_SOUP_DIR = "checkpoints/qwen4b_souped_fp16"
OUTPUT_SOUP_PATH = os.path.join(OUTPUT_SOUP_DIR, "classone_heads.pt")


def main():
    os.makedirs(OUTPUT_SOUP_DIR, exist_ok=True)
    heads_a = torch.load(HEADS_A_PATH, map_location="cpu")
    heads_b = torch.load(HEADS_B_PATH, map_location="cpu")

    alpha = 0.50  # Equal-weight model soup
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
    print(f"[✓] Created 50/50 souped heads at {OUTPUT_SOUP_PATH}")

    # Copy LoRA adapter to the soup directory as well
    src_lora = "checkpoints/qwen4b_fp16_candidate/lora_backbone"
    dst_lora = os.path.join(OUTPUT_SOUP_DIR, "lora_backbone")
    if os.path.exists(src_lora):
        if os.path.exists(dst_lora):
            shutil.rmtree(dst_lora)
        shutil.copytree(src_lora, dst_lora)
        print(f"[✓] Copied LoRA backbone to {dst_lora}")


if __name__ == "__main__":
    main()
