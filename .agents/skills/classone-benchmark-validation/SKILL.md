---
name: classone-benchmark-validation
description: >-
  Run, compare, and validate ClassOne model checkpoints or fresh Hugging Face Hub downloads
  against JevBench v1.6.1 (Easy, Original, Hard), Hugging Face Decision Index (5 domains),
  and RLCDAlignBench (10 failure modes). Automatically fixes Windows HF cache snapshot missing
  safetensors via NTFS hardlinks to bypass redundant 35GB re-downloads. Use when the user asks
  to validate models, run benchmarks, evaluate JevBench, Decision Index, or AlignBench, verify
  fresh HF downloads, compare candidate heads, or check for regressions.
---

# ClassOne Benchmark Validation Runbook

This skill provides the standard procedures and automated tools for evaluating ClassOne models across the official project benchmark triad and verifying downloaded Hugging Face checkpoints without network hangs or Windows cache failures.

---

## The ClassOne Benchmark Triad

Every ClassOne model candidate must be evaluated and verified across four complementary suites:

1. **JevBench Public Multi-Tier Benchmark (231 tasks)**:
   - **Easy Tier (48 tasks)**: Policy classification and basic refusal (`noul` + `choice`). Target: **100%**.
   - **Original Tier (72 tasks)**: Multi-turn dialog and scoring rubrics. Target: **>90%**.
   - **Hard Tier (111 tasks)**: Long-context legal policies and multi-hop constraint tracking. Target: **>45%**.
   - **Composite Score**: Harmonic/weighted benchmark heaven score across all tiers.
2. **Hugging Face Decision Index (50 tasks across 5 domains)**:
   - Evaluates Binary, Multiclass, and Scoring decisions across Customer Support, Healthcare/HIPAA, Legal, Security/Fintech, and Moderation.
   - Measures scale-invariant decision index ($100\%$ target).
3. **RLCDAlignBench (100 multi-battery instances across 10 failure modes)**:
   - Evaluates: Refusal, Sycophancy, Faithfulness, Privacy, Concealing Uncertainty, Reward Hacking, Power Seeking, Honesty (Deception), Bias, Prompt Injection.
   - Evaluates battery module strategies and inverts risk polarity for `deceptionbench_reward` (`risk = 1.0 - risk`).
   - Measures AUROC and Balanced Accuracy.
4. **NVIDIA When2Call Tool Routing (3,652 MCQ instances)**:
   - Evaluates action routing: `direct`, `tool_call`, `request_for_info`, and `cannot_answer`.
   - Evaluates single-pass tool invocation gating vs. token generation decode.
   - Runner: `scripts/eval_when2call.py --model <repo_id>`.
5. **Prompt Injection & Security Gating (deepset/prompt-injections)**:
   - Evaluates fast single-pass binary triage (`NoulQuestion`) on adversarial injections and jailbreaks.
   - Measures AUROC, Benign Specificity, and Injection Recall.
   - Runner: `scripts/eval_prompt_injection.py --model <repo_id>`.

---

## Critical Gotcha: Windows Hugging Face Cache Symlink Bug

### The Problem
On Windows systems without Windows Developer Mode enabled, Hugging Face Hub cannot create symlinks across snapshots in `E:\huggingface\hub` (or `~/.cache/huggingface/hub`). When a new commit updates only `classone_heads.pt` or `README.md`, HF Hub creates a new commit directory under `snapshots/<new_hash>/` that lacks the 18–35 GB `.safetensors` files. Calling `from_pretrained` then attempts to re-download 35 GB over the internet, causing long freezes and timeouts.

### The Automated Fix
Before running validation on a downloaded model, execute the cache fixer to link existing `.safetensors` using NTFS hardlinks (`os.link`):

```powershell
.venv\Scripts\python.exe scripts/fix_hf_cache_links.py --repo-id devops-thiago/classone-qwen3.5-9b
```
- **Cost**: 0 extra disk bytes (NTFS hardlink pointer).
- **Execution time**: ~0.02 seconds.
- You can also fix all cached ClassOne models at once with:
  ```powershell
  .venv\Scripts\python.exe scripts/fix_hf_cache_links.py
  ```

---

## Step-by-Step Validation Workflow

### Step 1: Pre-flight Verification & Environment Setup
Always ensure your terminal uses `.venv\Scripts\python.exe` and loads the Hugging Face token:

```powershell
# Verify GPU availability and memory
.venv\Scripts\python.exe -c "import torch; print(f'CUDA Devices: {torch.cuda.device_count()}, Device 0: {torch.cuda.get_device_name(0)}')"

# Fix HF cache snapshots to prevent redundant downloads
.venv\Scripts\python.exe scripts/fix_hf_cache_links.py --repo-id <REPO_ID>
```

### Step 2: Run Unified Benchmark Validation
Execute `scripts/run_hf_benchmark_validation.py` with `--include-alignbench`. This performs an atomic 3-benchmark run and SHA256 integrity verification:

```powershell
.venv\Scripts\python.exe scripts/run_hf_benchmark_validation.py `
  --repo-id devops-thiago/classone-qwen3.5-9b `
  --expected-head-sha e1f67a8c6418d20cca876b65a0ef323f0c93b26e81df277af909891d6bf5ea0d `
  --include-alignbench `
  --device cuda:0 `
  --pause-tokens 8
```

> [!IMPORTANT]
> Always pass `--include-alignbench` unless explicitly evaluating only JevBench. Without this flag, safety and alignment evaluation will be omitted.

### Step 3: Verify Results & Review Certificate
The script automatically writes a timestamped validation certificate to:
`benchmarks/results/live_hf_validation/<model_name>_live_hf_validation.json`

Check the generated certificate:
```powershell
Get-Content benchmarks/results/live_hf_validation/<model_name>_live_hf_validation.json
```

Verify that:
1. `head_sha256` matches the local snapshot candidate hash.
2. JevBench Easy tier is $\ge 98\%$.
3. JevBench Original tier is $\ge 90\%$.
4. Decision Index is $\ge 95\%$.
5. RLCDAlignBench Balanced Accuracy and AUROC meet or exceed published baselines.

---

## GPU Memory Hygiene During Sequential Evaluations

When evaluating 9B, 4B, and E4B sequentially on 16 GB RTX 5060 Ti GPUs:
- Always use 4-bit NF4 quantization (`quantization="4bit"`) in `from_backbone`.
- Run each model evaluation as a separate script invocation so Python and CUDA release all virtual memory allocations.
- If evaluating in a single script, ensure you call:
  ```python
  del model
  import gc
  gc.collect()
  torch.cuda.empty_cache()
  ```

---

## Reference Baselines (Project Records)

| Model | JevBench Overall | JevBench Easy | JevBench Orig | JevBench Hard | Decision Index | RLCDAlignBench BalAcc | Candidate Head SHA256 |
|---|---|---|---|---|---|---|---|
| **devops-thiago/classone-qwen3.5-9b** | **80.1% (185/231)** | **100.0%** | **98.6%** | **61.3%** | **100.0%** | **62.9% (0.629 AUROC)** | `e1f67a8c...` |
| **devops-thiago/classone-qwen3.5-4b** | **75.8% (175/231)** | **100.0%** | **95.8%** | **52.3%** | **98.99%** | **55.3% (0.569 AUROC)** | `d040fd3e...` |
| **devops-thiago/classone-gemma4-e4b** | **72.7% (168/231)** | **100.0%** | **95.8%** | **46.0%** | **97.16%** | **56.8% (0.560 AUROC)** | `42f9bb62...` |
| **devops-thiago/classone-qwen3.5-2b** | **69.7% (161/231)** | **100.0%** | **93.1%** | **41.4%** | **96.41%** | **54.1% (0.540 AUROC)** | `bd24d863...` |
| **devops-thiago/classone-gemma4-e2b** | **68.4% (158/231)** | **97.9%** | **90.3%** | **41.4%** | **95.80%** | **56.2% (0.562 AUROC)** | `3cb3a3fc...` |
