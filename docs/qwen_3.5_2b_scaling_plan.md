# Qwen 3.5 2B Scaling Plan & Blueprint

## 1. Executive Overview
- **Model Target:** `Qwen/Qwen3.5-2B` (~2.2B parameters, hybrid state-space/delta-rule architecture, $d_{\text{model}} = 2048$, 24 layers).
- **Objective:** Establish the first Qwen-family System 1 baseline for ClassOne, evaluating whether Qwen 3.5's hybrid linear attention achieves superior long-sequence efficiency and decision accuracy on JevBench and AlignBench.
- **Isolation Mandate:** This document and worktree (`worktrees/qwen-3.5-2b`) govern all Qwen 3.5 2B research, preserving the existing Gemma 2B, 4B, and 12B baselines untouched.

---

## 2. Hardware Budget & Feasibility (Dual NVIDIA RTX 5060 Ti - 16 GB Each)

Fine-tuning `Qwen/Qwen3.5-2B` in 4-bit NF4 precision with LoRA ($r=16, \alpha=32$) and gradient checkpointing:
- **Base Model (4-bit NF4):** ~1.80 GB VRAM.
- **Trainable LoRA + Decision Heads:** ~0.15 GB (~28M parameters).
- **Peak Training VRAM:** ~4.50 GB allocated per card.
- **Free Headroom Remaining:** **~11.50 GB free (72% margin)**.

---

## 3. Training & Evaluation Pipeline
1. **Curriculum:** 23,503-sample Champion corpus (`data/champion_70pct_final_corpus.jsonl`).
2. **Distributed Training:** Dual-GPU shared-memory IPC, linear warmup, cosine decay.
3. **Benchmarks:** 231 JevBench public tasks and 100 RLCDAlignBench instances.
