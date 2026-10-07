#!/usr/bin/env python3
"""Performs full transformer backbone and decision head model souping (averaging)
between classone_enterprise_boost_merged (52/111 Hard) and classone_sota_70pct_final_merged (62/72 Original)."""

import os

import torch
from transformers import AutoModel, AutoTokenizer

MODEL_1_PATH = "checkpoints/classone_enterprise_boost_merged"
MODEL_2_PATH = "checkpoints/classone_sota_70pct_final_merged"

HEADS_1_PATH = "checkpoints/classone_enterprise_boost/classone_heads.pt"
HEADS_2_PATH = "checkpoints/classone_sota_70pct_final/classone_heads.pt"

OUTPUT_MODEL_DIR = "checkpoints/classone_model_soup_70_merged"
OUTPUT_HEADS_PATH = "checkpoints/classone_model_soup_70/classone_heads.pt"


def main():
    print(f"[*] Starting full Model Soup between {MODEL_1_PATH} and {MODEL_2_PATH}...")
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_HEADS_PATH), exist_ok=True)

    # 1. Soup Decision Heads
    print("[*] Souping decision heads (alpha=0.50)...")
    h1 = torch.load(HEADS_1_PATH, map_location="cpu")
    h2 = torch.load(HEADS_2_PATH, map_location="cpu")

    soup_heads = {}
    for hname in ["noul_head", "choice_head", "score_head"]:
        soup_heads[hname] = {}
        for k in h1[hname]:
            if h1[hname][k].is_floating_point():
                soup_heads[hname][k] = 0.50 * h1[hname][k] + 0.50 * h2[hname][k]
            else:
                soup_heads[hname][k] = h1[hname][k]

    torch.save(soup_heads, OUTPUT_HEADS_PATH)
    print(f"[✓] Saved souped heads to {OUTPUT_HEADS_PATH}")

    # 2. Soup Transformer Backbone Weights on CPU
    print("[*] Loading backbone weights for Model 1...")
    m1 = AutoModel.from_pretrained(MODEL_1_PATH, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True)
    state_m1 = m1.state_dict()

    print("[*] Loading backbone weights for Model 2...")
    m2 = AutoModel.from_pretrained(MODEL_2_PATH, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True)
    state_m2 = m2.state_dict()

    print("[*] Interpolating backbone weights (50% Enterprise Boost + 50% SOTA Final)...")
    souped_state = {}
    for k, v1 in state_m1.items():
        if k in state_m2:
            v2 = state_m2[k]
            if v1.is_floating_point():
                souped_state[k] = 0.50 * v1 + 0.50 * v2
            else:
                souped_state[k] = v1
        else:
            souped_state[k] = v1

    # Clean up m2 from memory
    del m2, state_m2

    print("[*] Loading souped weights into model...")
    m1.load_state_dict(souped_state)
    del souped_state

    print(f"[*] Saving souped model shards to {OUTPUT_MODEL_DIR}...")
    m1.save_pretrained(OUTPUT_MODEL_DIR, safe_serialization=True, max_shard_size="5GB")
    del m1

    # Copy tokenizer
    from classone.tokenizer import ClassOnePromptBuilder

    tok = AutoTokenizer.from_pretrained(MODEL_2_PATH)
    ClassOnePromptBuilder(tok)
    tok.save_pretrained(OUTPUT_MODEL_DIR)

    print(f"[✓] Full Model Soup complete! Merged model saved to {OUTPUT_MODEL_DIR}")


if __name__ == "__main__":
    main()
