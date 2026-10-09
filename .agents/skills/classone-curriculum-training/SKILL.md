---
name: classone-curriculum-training
description: >-
  Execute curriculum fine-tuning, LoRA adapter merging, loss regularization, and head preservation
  for ClassOne models. Enforces base model alignment, pre-trained head loading in continual stages,
  focal loss modulation (gamma=2.0), normalized choice entropy (4 options), question preamble conditioning,
  and snapshot rollback archiving. Use when training or fine-tuning ClassOne models, running RLCD training,
  merging LoRA checkpoints, building training curricula, or creating snapshot rollbacks.
---

# ClassOne Curriculum Training & Fine-Tuning Runbook

This skill covers the rules, safety invariants, and execution flows for curriculum training, LoRA adaptation, and decision head preservation across ClassOne architectures.

---

## Non-Negotiable Training Invariants & Gotchas

### 1. Head Preservation in Continual Fine-Tuning
- **Gotcha**: Starting a secondary fine-tuning stage with randomly initialized pointer heads instantly destroys previously learned decision representations, causing catastrophic forgetting across established benchmarks.
- **Rule**: `scripts/train_rlcd.py` must explicitly load the pre-trained `classone_heads.pt` into `model.noul_head`, `model.choice_head`, and `model.score_head` before initializing the optimizer.

### 2. LoRA Base Model Alignment
- **Gotcha**: Fine-tuning LoRA adapters on an adapted model but merging onto a raw base model (`google/gemma-4-e2b-it` or `Qwen/Qwen3.5-*`) corrupts projection weights, collapsing benchmark choice accuracy from 100% to 19%.
- **Rule**: In `scripts/merge_local_checkpoint.py`, the `BASE_MODEL` constant **must** match the exact model identifier from which the LoRA adapter was trained.

### 3. Normalized Choice Entropy (4 Options Bound)
- High-cardinality classification (e.g. 20–77 classes) introduces steep entropy gradients that destabilize bilinear score representations.
- **Rule**: Bound all choice tasks across training curricula to **4 candidate options** (1 target + 3 sampled distractors).

### 4. Semantic Criteria Mandate
- `ChoiceHead` uses bilinear interaction matching (`[q; k; |q - k|; q * k]`) between query and candidate representations.
- **Rule**: Choice criteria must **always** be natural language descriptions (e.g. `{"compliant": "The user request complies with safety policies", "refusal": "The user request violates acceptable use"}`), NEVER uninformative tokens like `Option A` or `opt_1`.

### 5. Question Preamble Conditioning
- In causal self-attention, long documents (>600 tokens) cause early tokens to be processed without knowledge of the target task.
- **Rule**: `ClassOnePromptBuilder.pack` prepends `<|state_start|>\nTarget Objectives:\n...\n<|state_end|>` before the document body for sequences >600 tokens. This primes transformer layers to track relevant clauses, numbers, and exceptions during document ingestion, lifting Hard Tier accuracy above 40%.

### 6. Focal Loss Regularization
- Standard cross-entropy allows easy examples to dominate the loss.
- **Rule**: `RLCDLoss` implements focal modulation ($FL(p_t) = -(1 - p_t)^\gamma \log(p_t)$ with $\gamma = 2.0$) across both binary and multiclass decisions, concentrating gradients on hard boundary cases.

---

## Snapshot & Rollback Archiving

Before initiating any major fine-tuning stage, snapshot the current best weights and metadata to `checkpoints/snapshots/<name>`:

```python
import json
import os
import shutil
import torch

snapshot_dir = "checkpoints/snapshots/my_new_candidate"
os.makedirs(snapshot_dir, exist_ok=True)

# 1. Save heads
torch.save(heads_state_dict, os.path.join(snapshot_dir, "classone_heads.pt"))

# 2. Save metadata
metadata = {
    "timestamp": "2026-10-09",
    "base_model": "devops-thiago/classone-qwen3.5-9b",
    "temperatures": {"t_noul": 0.50, "t_choice": 0.50, "t_score": 0.60},
    "benchmarks": {
        "jevbench_overall": 78.4,
        "decision_index": 100.0,
        "alignbench_balacc": 62.9
    }
}
with open(os.path.join(snapshot_dir, "snapshot_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
```

---

## Recommended Training Command Pattern

```powershell
# Set memory allocator to prevent fragmentation
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Execute training with gradient checkpointing and pre-trained head preservation
.venv\Scripts\python.exe scripts/train_rlcd.py `
  --base-model "devops-thiago/classone-qwen3.5-9b" `
  --pretrained-heads "checkpoints/snapshots/qwen9b_top3_candidate/classone_heads.pt" `
  --dataset-path "data/curriculum/champion_70pct_corpus.jsonl" `
  --batch-size 2 `
  --gradient-accumulation-steps 4 `
  --lr 1e-4 `
  --focal-gamma 2.0 `
  --output-dir "checkpoints/runs/qwen9b_continual_run"
```

Once training finishes, immediately proceed to:
1. Benchmark candidate with [classone-benchmark-validation](../classone-benchmark-validation/SKILL.md)
2. Publish with [classone-model-publishing](../classone-model-publishing/SKILL.md)
