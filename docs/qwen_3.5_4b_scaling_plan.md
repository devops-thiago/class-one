# Qwen 3.5 4B Scaling Plan & Blueprint

## 1. Executive Overview
- **Model Target:** `Qwen/Qwen3.5-4B` (~4.5B parameters, hybrid state-space/delta-rule architecture, $d_{\text{model}} = 2560$, 32 layers).
- **Objective:** Evaluate Qwen 3.5 at the 4B scale with 2560 hidden dimension and 32 layers against Gemma 4 E4B, evaluating trade-offs between linear state-space attention and standard causal self-attention.
- **Isolation Mandate:** This document and worktree (`worktrees/qwen-3.5-4b`) govern all Qwen 3.5 4B research.

---

## 2. Hardware Budget & Feasibility (Dual NVIDIA RTX 5060 Ti - 16 GB Each)

Fine-tuning `Qwen/Qwen3.5-4B` in 4-bit NF4 precision with LoRA ($r=16, \alpha=32$) and gradient checkpointing:
- **Base Model (4-bit NF4):** ~2.90 GB VRAM.
- **Trainable LoRA + Decision Heads:** ~0.20 GB (~35M parameters).
- **Peak Training VRAM:** ~5.80 GB allocated per card.
- **Free Headroom Remaining:** **~10.20 GB free (64% margin)**.

---

## 3. Training & Evaluation Pipeline
1. **Curriculum:** 23,503-sample Champion corpus (`data/champion_70pct_final_corpus.jsonl`).
2. **Distributed Training:** Dual-GPU shared-memory IPC, linear warmup, cosine decay.
3. **Benchmarks:** 231 JevBench public tasks and 100 RLCDAlignBench instances.
