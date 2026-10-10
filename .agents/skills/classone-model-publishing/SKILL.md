---
name: classone-model-publishing
description: >-
  Prepare, verify, and publish ClassOne candidate models, decision heads, tokenizers,
  and model cards to Hugging Face Hub (devops-thiago/classone-*). Enforces fast atomic
  uploads (--update-heads-only), delimiter token preservation (<|pause|>, Gemma 4 PLE unused mapping),
  head SHA256 integrity checks, and benchmark card synchronization. Use when the user asks
  to push a model to HF, publish candidate heads, update model card README, sync tokenizers,
  or upload new checkpoints to Hugging Face.
---

# ClassOne Model Publishing Runbook

This skill provides step-by-step procedures and automated helpers for publishing ClassOne models and candidate heads to Hugging Face Hub safely, quickly, and with full architectural integrity.

---

## Publishing Principles & Gotchas

### 1. Publishing Standard: Unquantized Safetensors Only
- Quantized GGUF models are deprecated. Push **only** the unquantized model (`safetensors` shards + calibrated `classone_heads.pt` + tokenizer + config + `README.md`) to Hugging Face Hub.

### 2. Fast Atomic Publishing (`--update-heads-only`)
- **Gotcha**: Uploading 35 GB of safetensors files via standard `upload_folder` hashes every tensor file on disk. This takes 10–20 minutes and frequently drops connections.
- **Solution**: When updating heads, tokenizers, configs, or README, always pass `--update-heads-only` to `scripts/push_to_hub.py`. It only uploads modified lightweight artifacts, publishing in **~5 seconds**.

### 3. Special Delimiter Tokens Serialization
- Standalone merged model tokenizers must always pass through `ClassOnePromptBuilder(tokenizer)` before `tokenizer.save_pretrained()` so custom delimiter tokens (`<|state_start|>`, `<|state_end|>`, `<|pause|>`, etc.) are serialized into `tokenizer.json`.
- **Gemma 4 Per-Layer Embeddings (PLE) Rule**: Delimiter tokens **must** map to in-vocab `<unused0>`...`<unused12>`. Do **never** add new special tokens to Gemma 4 tokenizers, as resizing vocab across 42 PLE layers causes an immediate 5.25 GB VRAM explosion and embedding mismatch.
- **Qwen Rule**: For Qwen 3.5, delimiter tokens (including `<|pause|>`) are added as special tokens without corrupting chat template.

### 4. Calibrated Temperature Floor Clamping
- Never allow post-hoc calibration to push temperatures $> 1.0$.
- Temperatures must be clamped to $T_{\text{noul}}, T_{\text{choice}} \in [0.45, 0.60]$ and $T_{\text{score}} \in [0.55, 0.65]$.
- Temperatures $> 1.0$ flatten probability margins between candidate options, collapsing choice accuracy on fine-grained benchmarks.

---

## Step-by-Step Publishing Workflow

### Step 1: Stage Candidate Head & Serialize Delimiters
Use the automated helper `scripts/prepare_candidate_for_hub.py`:

```powershell
.venv\Scripts\python.exe scripts/prepare_candidate_for_hub.py `
  --snapshot-dir "checkpoints/snapshots/qwen9b_top3_candidate" `
  --target-dir "checkpoints/classone_qwen9b_merged" `
  --repo-id "devops-thiago/classone-qwen3.5-9b" `
  --pause-tokens 8
```

This helper will:
1. Verify `classone_heads.pt` exists and compute its SHA256.
2. Confirm calibrated temperatures are $\le 1.0$.
3. Run `ClassOnePromptBuilder` and save `tokenizer.json` with delimiters properly registered.
4. Copy `classone_heads.pt` to the staging directory.
5. Print the exact atomic upload command.

### Step 2: Update Model Card (`README.md`)
Update the model card in the target directory with:
- Updated JevBench tier accuracy table (Easy, Original, Hard, and Composite).
- HF Decision Index score ($100\%$).
- RLCDAlignBench Balanced Accuracy and AUROC.
- Certified candidate head SHA256.

### Step 3: Perform Fast Atomic Push
Execute `scripts/push_to_hub.py` with `--update-heads-only`:

```powershell
.venv\Scripts\python.exe scripts/push_to_hub.py `
  --checkpoint-dir "checkpoints/classone_qwen9b_merged" `
  --repo-id "devops-thiago/classone-qwen3.5-9b" `
  --update-heads-only
```

### Step 4: Verify Live Remote Commit
Confirm the remote repository has updated:
```powershell
.venv\Scripts\python.exe -c "from huggingface_hub import HfApi, hf_hub_download; import hashlib; p = hf_hub_download('devops-thiago/classone-qwen3.5-9b', 'classone_heads.pt', force_download=True); print('Remote Head SHA:', hashlib.sha256(open(p, 'rb').read()).hexdigest())"
```

Next, immediately proceed to run the validation runbook:
[classone-benchmark-validation](../classone-benchmark-validation/SKILL.md)
