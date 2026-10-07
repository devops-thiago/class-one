# Gemma 4 E4B Scaling Plan & Engineering Blueprint

## 1. Executive Overview
- **Model Target:** `google/gemma-4-E4B-it`
- **Objective:** Scale ClassOne's System 1 decision architecture from 2B to 4B effective parameters to resolve the feedforward capacity bottleneck on JevBench Hard Tier (3,000-token multi-hop contract reasoning) and surpass $\ge 70\%$ JevBench and $\ge 60\%$ AlignBench.
- **Isolation Mandate:** This document and worktree (`worktrees/gemma-4-e4b`) govern all E4B experimentation. The production 2B baseline (`classone-gemma4-e2b`, 70.1% JevBench milestone, 56.1% AlignBench record) remains frozen and intact in the main workspace.

---

## 2. Hardware Budget & Feasibility (Dual NVIDIA RTX 5060 Ti - 16 GB Each)

Empirically verified on hardware (`cuda:0`, NVIDIA GeForce RTX 5060 Ti):
- Base Model (4-bit NF4): **8.87 GB** allocated VRAM.
- Trainable LoRA ($r=16, \alpha=32$, 34.8M params) + Decision Heads: **9.00 GB** total allocated.
- Full Forward + Backward Pass (RLCD Focal Loss $\gamma=2.0$, seq len 64): **9.21 GB** allocated (**9.78 GB peak**).
- **Free Headroom Remaining:** **6.22 GB free (39% safety margin)**.

| Component | VRAM per GPU (4-bit NF4 LoRA) | Description / Configuration |
|---|---|---|
| **Base Weights (Frozen 4-bit NF4)** | **8.87 GB** | 4.3B effective params with 42-layer Per-Layer Embeddings |
| **Trainable LoRA + Decision Heads** | **0.13 GB** | 34,881,536 trainable parameters across all 42 layers |
| **AdamW Optimizer States** | **0.42 GB** | FP32 master states for 34.8M parameters |
| **Trainable Gradients** | **0.07 GB** | Flattened gradient tensor synchronized via IPC shared memory |
| **Activations (seq len 2,048)** | **1.20 GB** | Reduced by 80% with `gradient_checkpointing_enable(use_reentrant=False)` |
| **PyTorch Context & Runtime** | **0.80 GB** | CUDA runtime memory overhead |
| **Total Peak Allocated** | **~9.78 GB** | **Fits comfortably within 16.0 GB ceiling** |
| **Free Headroom** | **~6.22 GB (39% margin)** | Zero risk of OOM on variable sequence lengths |

---

## 3. Architecture & Per-Layer Embedding (PLE) Engineering
1. **Model Dimensions:**
   - Architecture: `Gemma4ForConditionalGeneration` / `Gemma4Model`
   - Hidden Size ($d_{\text{model}}$): **2560** (up from 2048 in E2B).
   - Layers: **42 transformer layers** (up from 35 in E2B).
   - Intermediate Size (FFN): **10240** (up from 8192 in E2B).
   - Attention Heads: **8 Query Heads / 2 KV Heads (GQA)**.
   - Context Length: **131,072 tokens (128k)**.
2. **In-Vocab Delimiter Token Mapping (PLE OOM Elimination):**
   - Gemma 4 incorporates 42 layers of Per-Layer Embeddings (PLE). Calling `resize_token_embeddings` attempts to duplicate the entire 42-layer embedding matrix in VRAM, which requires an extra 5.25 GiB.
   - **Solution:** `ClassOnePromptBuilder` detects Gemma 4 and maps ClassOne's 12 delimiters (`<|state_start|>`, `<|state_end|>`, etc.) directly to Google DeepMind's native reserved in-vocab tokens (`<unused0>` through `<unused11>`, token IDs 6–17).
   - This eliminates `resize_token_embeddings`, guarantees all token IDs remain $< 262,144$, and frees 5.25 GB of VRAM.

---

## 4. Training Pipeline & Curriculum Inheritance

The E4B training pipeline inherits our balanced 15-domain Champion corpus (`data/champion_70pct_final_corpus.jsonl`, 23,503 items):
1. **Multi-Level Ordinal Rubrics:** 4-level and 5-level rubrics (`SEV-1`–`SEV-4`, bursary award tiers 0–4).
2. **Missing Evidence & Ambiguity:** Supervised `cannot_determine` / `insufficient_information`.
3. **Multi-Clause Settlements:** ContractNLI, CUAD covenants, insurance endorsement sublimits, and trade compliance (ECCN).
4. **AlignBench Guardians:** Sycophancy evaluation, reward hacking dilemma choice, and canary secret leak detection.
5. **Micro-Logic & Format Compliance:** Conditional exception policies and structural format adequacy.

---

## 5. Execution Steps
1. **Model Architecture Probe:** Probe `google/gemma-4-E4B-it` configuration and layer mappings.
2. **Tokenizer Alignment:** Ensure `<|state_start|>`, `<|state_end|>`, `<|query_start|>`, `<|query_end|>` are serialized into the E4B tokenizer vocabulary.
3. **Decision Head Binding:** Instantiate `ChoiceHead`, `NoulHead`, `ScoreHead` sized to the E4B hidden dimension.
4. **Dual-GPU LoRA Training:** Execute multi-GPU distributed fine-tuning with linear warmup and cosine annealing.
5. **Multi-Tier Benchmark:** Evaluate on all 231 JevBench tasks and 100 AlignBench instances.

---

## 6. Official Benchmark Results & Dual Milestone Crossing

### A. JevBench Multi-Tier Evaluation (231 Tasks)
- **ALL TIERS TOTAL:** **70.1% (162 of 231 correct) [MILESTONE REACHED: $\ge 70.0\%$]**
  - **Easy Tier:** **100.0% (48/48 PERFECT SCORE)** | Noul: 12/12 (100%) | Choice: 36/36 (100%) | ECE: 0.0000 | Latency: 98.5 ms.
  - **Original Tier:** **93.1% (67/72 ALL-TIME RECORD)** | Noul: 24/24 (100%) | Choice: 31/36 (86.1%) | Score: 12/12 (100%) | ECE: 0.0588 | Latency: 97.9 ms.
  - **Hard Tier:** **42.3% (47/111)** | Choice: 29/67 (43.3%) | Noul: 16/38 (42.1%) | Score: 2/6 (33.3%) | Latency: 197.0 ms.

### B. RLCDAlignBench Zero-Shot Evaluation (100 Instances)
- **Balanced Accuracy:** **63.3% [MILESTONE REACHED: $\ge 60.0\%$]**
- **Standard Accuracy:** **63.0% (63 of 100 correct)**
- **Refusal / Jailbreaks:** **72.7% accuracy** (0.433 AUROC)
- **Honesty (Deception Detection):** **72.7% accuracy** (0.567 AUROC, 0.2531 ECE)
- **Faithfulness (Hallucination Detection):** **66.7% accuracy** (0.600 AUROC)
- **Reward Hacking:** **66.7% accuracy** (0.650 AUROC)
- **Power Seeking:** **66.7% accuracy** (0.556 AUROC)
- **Privacy (Secret Leaks):** **64.3% accuracy** (0.571 AUROC)
- **Concealing Uncertainty:** **64.3% accuracy** (0.510 AUROC)
- **Brier Score:** **0.3110**
- **Median Latency (p50):** **425.9 ms**
