#!/usr/bin/env python3
"""CLI utility to publish a trained ClassOne model checkpoint to the Hugging Face Hub.

Two publication modes:

1. Adapter mode (default) — uploads ``classone_heads.pt`` plus the ``lora_backbone/``
   LoRA adapter and a model card. The consumer must download the Gemma base model
   separately and attach the adapter.
2. Standalone mode (``--merge-standalone``) — merges the LoRA adapter into the base
   weights and publishes a fully self-contained model (weights + tokenizer + config)
   to the repository root, so the model loads as a single Hugging Face model.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

from huggingface_hub import HfApi, create_repo

from classone.publishing import (
    check_candidate_regression,
    get_registered_heads_sha256,
    load_baseline_registry,
    validate_markdown_attribution,
)


def _infer_base_model(repo_id: str) -> str:
    """Infers canonical base model identifier from repository name."""
    rid = repo_id.lower()
    if "qwen" in rid and "9b" in rid:
        return "Qwen/Qwen3.5-9B"
    if "qwen" in rid and "4b" in rid:
        return "Qwen/Qwen3.5-4B"
    if "qwen" in rid and "2b" in rid:
        return "Qwen/Qwen3.5-2B"
    if "e4b" in rid:
        return "google/gemma-4-E4B-it"
    return "google/gemma-4-E2B-it"


def _get_git_commit_sha() -> str:
    """Returns short git commit SHA of the current workspace HEAD, or 'unknown'."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def parse_args():
    parser = argparse.ArgumentParser(description="Publish ClassOne model to Hugging Face Hub")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Path to trained checkpoint directory (containing classone_heads.pt and lora_backbone/). "
        "Required for adapter mode; optional in standalone mode (adapter is then read from the Hub repo).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Target HF repository ID (e.g. username/classone-gemma4-e2b)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model ID used during training (auto-inferred from --repo-id if omitted)",
    )
    parser.add_argument(
        "--merge-standalone",
        action="store_true",
        help="Merge the LoRA adapter into the base model and publish a self-contained model "
        "(weights + tokenizer + config) to the repository root.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Precision used while merging the adapter (default: bfloat16)",
    )
    parser.add_argument(
        "--merged-dir",
        type=str,
        default=None,
        help="Directory holding a merged standalone model tree. If it already contains weights, "
        "they are uploaded as-is; otherwise the adapter is merged into this directory first. "
        "Requires --merge-standalone.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device used to merge the adapter (default: auto-detects cuda > mps > cpu)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (defaults to HF_TOKEN env var)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private",
    )
    parser.add_argument(
        "--update-heads-only",
        action="store_true",
        help="Update decision heads, tokenizer, and model card without re-uploading base weights",
    )
    parser.add_argument(
        "--readme-only",
        action="store_true",
        help="Regenerate and upload only README.md to the Hub without touching weights or classone_heads.pt",
    )
    parser.add_argument(
        "--commit-sha",
        type=str,
        default=None,
        help="Git commit SHA to embed in model card self-measurement metadata (defaults to HEAD)",
    )
    parser.add_argument(
        "--eval-report",
        type=str,
        default=None,
        help="Path to candidate evaluation JSON report verified against benchmarks/baselines.json before upload.",
    )
    parser.add_argument(
        "--allow-regression",
        action="store_true",
        help="Override non-regression baseline gate even if candidate metrics fall below benchmarks/baselines.json.",
    )
    parser.add_argument(
        "--skip-eval-gate",
        action="store_true",
        help="Explicitly bypass requiring --eval-report when uploading weights or heads.",
    )
    return parser.parse_args()


def _resolve_dtype(name: str):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _resolve_device(name: str) -> str:
    import torch

    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_card_provenance_defaults() -> tuple[str, str]:
    """Reads canonical evaluation date and hardware tag from benchmarks/baselines.json."""
    try:
        registry = load_baseline_registry()
        return str(registry.get("eval_date", "2026-11-13")), str(
            registry.get("hardware", "2x NVIDIA RTX 5060 Ti 16 GB")
        )
    except OSError:
        return "2026-11-13", "2x NVIDIA RTX 5060 Ti 16 GB"


def generate_model_card(
    repo_id: str,
    base_model: str,
    standalone: bool = False,
    commit_sha: str | None = None,
) -> str:
    """Generates a Hugging Face model card with verified self-measured and official benchmark results."""
    short_name = repo_id.split("/")[-1]
    sha = commit_sha or _get_git_commit_sha()
    eval_date, hardware_tag = _load_card_provenance_defaults()
    certified_heads_sha = get_registered_heads_sha256(repo_id)
    heads_sha_line = (
        f"\n> **Certified `classone_heads.pt` SHA256:** `{certified_heads_sha}`\n" if certified_heads_sha else ""
    )
    dtype_expr = "torch.bfloat16" if any(k in repo_id.lower() for k in ("9b", "4b", "e4b")) else "torch.float16"
    is_qwen = "qwen" in base_model.lower() or "qwen" in repo_id.lower()
    arch_tag = "qwen" if is_qwen else "gemma"

    if standalone:
        intro = (
            f"**[{repo_id}](https://huggingface.co/{repo_id})** is an open-source **System 1 decision model** "
            f"built on [`{base_model}`](https://huggingface.co/{base_model}) "
            f"using the [ClassOne (Classifier One) architecture](https://github.com/devops-thiago/class-one). "
            f"The merged standalone backbone ships directly in this repository — it loads as a single model, "
            f"with no adapter and no separate base-model download."
        )
        quickstart = f'''```python
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from classone.modeling.modeling_classone import ClassOneModel
from classone.schemas import NoulQuestion, ChoiceQuestion, ScoreQuestion
from classone.tokenizer import ClassOnePromptBuilder

REPO_ID = "{repo_id}"

# 1. Load the ClassOne model (weights + tokenizer are fully self-contained here)
tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
builder = ClassOnePromptBuilder(tokenizer)
model = ClassOneModel.from_backbone(
    base_model_name_or_path=REPO_ID,
    tokenizer=tokenizer,
    device="cuda",
    torch_dtype={dtype_expr},
)

# 2. Load the trained decision heads
heads = torch.load(hf_hub_download(REPO_ID, "classone_heads.pt"), map_location="cuda")
model.noul_head.load_state_dict(heads["noul_head"], strict=False)
model.choice_head.load_state_dict(heads["choice_head"], strict=False)
model.score_head.load_state_dict(heads["score_head"], strict=False)
model.eval()

# 3. Pack state + questions and run a single forward pass
packed = builder.pack(
    state={{"customer": "Alex", "message": "I was charged twice for order #123."}},
    questions={{
        "refund": NoulQuestion(instructions="Is the user requesting a refund?"),
        "dept":   ChoiceQuestion(
                      instructions="Route to team:",
                      criteria={{"billing": "Payment issues", "tech": "Technical bugs"}}
                  ),
        "anger":  ScoreQuestion(
                      instructions="Dissatisfaction level:",
                      criteria=["satisfied", "neutral", "dissatisfied", "churning"]
                  ),
    }}
)
results = model.evaluate_packed(packed)

print("Refund P(true):", results["refund"].noul)
print("Department:    ", results["dept"].choice, "—", results["dept"].probabilities)
print("Anger score:   ", results["anger"].score)
```'''
        files_table = """| File | Description |
|---|---|
| `model.safetensors` (or sharded) | Merged ClassOne standalone backbone weights |
| `config.json` | Model configuration |
| `tokenizer.json`, `tokenizer_config.json` | Tokenizer with ClassOne delimiter tokens |
| `classone_heads.pt` | Trained Noul / Choice / Score head weights + calibrated temperatures |"""
    else:
        intro = (
            f"**[{repo_id}](https://huggingface.co/{repo_id})** is an open-source **System 1 decision model** "
            f"built on top of [`{base_model}`](https://huggingface.co/{base_model}) "
            f"using the [ClassOne architecture](https://github.com/devops-thiago/class-one)."
        )
        quickstart = f'''```python
import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoTokenizer

from classone.modeling.modeling_classone import ClassOneModel
from classone.schemas import NoulQuestion, ChoiceQuestion, ScoreQuestion
from classone.tokenizer import ClassOnePromptBuilder

REPO_ID = "{repo_id}"

# 1. Load the backbone and attach the published LoRA adapter
tokenizer = AutoTokenizer.from_pretrained("{base_model}")
builder = ClassOnePromptBuilder(tokenizer)
model = ClassOneModel.from_backbone(
    base_model_name_or_path="{base_model}",
    tokenizer=tokenizer,
    device="cuda",
    torch_dtype={dtype_expr},
)
model.backbone = PeftModel.from_pretrained(model.backbone, REPO_ID, subfolder="lora_backbone")

# 2. Load the trained decision heads
heads = torch.load(hf_hub_download(REPO_ID, "classone_heads.pt"), map_location="cuda")
model.noul_head.load_state_dict(heads["noul_head"], strict=False)
model.choice_head.load_state_dict(heads["choice_head"], strict=False)
model.score_head.load_state_dict(heads["score_head"], strict=False)
model.eval()

# 3. Pack state + questions and run a single forward pass
packed = builder.pack(
    state={{"customer": "Alex", "message": "I was charged twice for order #123."}},
    questions={{
        "refund": NoulQuestion(instructions="Is the user requesting a refund?"),
        "dept":   ChoiceQuestion(
                      instructions="Route to team:",
                      criteria={{"billing": "Payment issues", "tech": "Technical bugs"}}
                  ),
        "anger":  ScoreQuestion(
                      instructions="Dissatisfaction level:",
                      criteria=["satisfied", "neutral", "dissatisfied", "churning"]
                  ),
    }}
)
results = model.evaluate_packed(packed)

print("Refund P(true):", results["refund"].noul)
print("Department:    ", results["dept"].choice, "—", results["dept"].probabilities)
print("Anger score:   ", results["anger"].score)
```'''
        files_table = """| File | Description |
|---|---|
| `classone_heads.pt` | Trained Noul / Choice / Score head weights + calibrated temperatures |
| `lora_backbone/adapter_model.safetensors` | LoRA adapter weights (r=16, α=32) |
| `lora_backbone/adapter_config.json` | LoRA config (target modules, rank, etc.) |"""

    is_qwen9b = "qwen" in repo_id.lower() and "9b" in repo_id.lower()
    is_qwen4b = "qwen" in repo_id.lower() and "4b" in repo_id.lower()
    is_qwen2b = "qwen" in repo_id.lower() and "2b" in repo_id.lower()
    is_e4b = "e4b" in repo_id.lower()

    provenance_banner = (
        f"> **Self-Measured Evaluation Provenance:** Executed locally on `{hardware_tag}` "
        f"(Date: `{eval_date}`, Commit: `{sha}`). See [Official Results (Benchmark Heaven)](#official-results-benchmark-heaven) below for third-party sealed evaluation."
    )
    table_meta = f"*Self-measured on `{hardware_tag}` · Date: `{eval_date}` · Commit: `{sha}`*"

    if is_qwen9b:
        official_section = f"""## Official Results (Benchmark Heaven)

Official third-party evaluation on **[Benchmark Heaven](https://benchmarkheaven.com/jev-models)** ([model page: `classone-qwen3.5-9b`](https://benchmarkheaven.com/jev-models/classone-qwen3.5-9b), benchmark version `v1.8.8`, checked `{eval_date}`):

| Metric | Official Benchmark Heaven (`v1.8.8`) |
|---|---|
| **Composite Score** | **43.0** (Composite rank **#131**) |
| **Intelligence** | **42.2** |
| **Calibration** | **58.7** |
| **Capability** | **50.5** (rank **#181** of 243 open-weights models) |

**Why self-measured local numbers differ from official Benchmark Heaven scores:**
1. **Sealed vs. public split:** Local self-measurements evaluate strictly the **231 public JevBench tasks**, whereas Benchmark Heaven evaluates both public tasks and private **sealed items**.
2. **Public/sealed gap penalty:** Benchmark Heaven applies a generalization penalty when performance diverges between public and sealed subsets.
3. **Strict 10 s rule & latency/cost normalization:** Tasks exceeding 10 seconds are penalized to zero, and speed/cost axes are normalized on server-grade reference GPUs rather than local workstation hardware (`{hardware_tag}`)."""
        benchmark_section = f"""{provenance_banner}

### 1. JevBench Public Multi-Tier Suite (231 Public Tasks, Self-Measured)

Evaluated standalone on the 231 public tasks in [fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench) (heads SHA256 `68FCDB31...`).

{table_meta}

| Tier | Tasks | Accuracy | ECE | Brier Score | Median Latency (p50) |
|---|---|---|---|---|---|
| **Easy** | 48 | **100.0%** (48/48) | **0.0029–0.0031** | **0.0000–0.0001** | **112.8 ms** |
| **Original** | 72 | **97.2%** (70/72) | **0.0057–0.0059** | **0.0209** | **111.5 ms** |
| **Hard** | 111 | **59.5%** (66/111) | 0.2537 | 0.2946 | **320.1 ms** |
| **Overall Aggregate (Public)** | **231** | **79.7%** (184/231) | — | — | **112.8 ms** |

### 2. Decision Index Suite (Self-Run, 50 Tasks)

Self-run local evaluation across the 50 tasks from [`multimodalart/jev-decision-index`](https://huggingface.co/datasets/multimodalart/jev-decision-index).

{table_meta}

| Domain | Tasks | Accuracy (%) |
|---|---|---|
| **Knowledge** | 10 | **100.0%** (10/10) |
| **Language** | 10 | **100.0%** (10/10) |
| **Retrieval** | 10 | **100.0%** (10/10) |
| **Rubric** | 10 | **100.0%** (10/10) |
| **Tools** | 10 | **100.0%** (10/10) |
| **Overall Suite Score** | **50** | **100.0%** (50/50, 0.0138–0.0152 ECE) |

### 3. RLCDAlignBench Alignment & Safety Suite (100 Instances, Self-Measured)

Self-measured across the 10 alignment failure modes in [RLCDAlignBench](https://huggingface.co/datasets/sumleo/RLCDAlignBench) (`arXiv:2609.29429`).

{table_meta}

| Failure Mode / Axis | Samples (N) | AUROC | Accuracy (%) | ECE | Latency (p50) |
|---|---|---|---|---|---|
| **Honesty (Deception)** | 11 | **0.933** | **81.8%** | 0.2067 | 756.0 ms |
| **Refusal (Jailbreaks)** | 11 | **0.867** | **81.8%** | 0.2240 | 901.0 ms |
| **Faithfulness** | 9 | **0.800** | **66.7%** | 0.2328 | 729.0 ms |
| **Power Seeking** | 6 | **0.778** | **83.3%** | 0.1912 | 716.7 ms |
| **Concealing Uncertainty** | 14 | **0.776** | **85.7%** | 0.3477 | 448.8 ms |
| **Social Bias** | 9 | **0.675** | **55.6%** | 0.4213 | 738.2 ms |
| **Prompt Injection** | 8 | **0.625** | **50.0%–62.5%** | 0.4695 | 584.6 ms |
| **Privacy** | 14 | **0.612** | **50.0%** | 0.3584 | 570.0 ms |
| **Sycophancy** | 9 | 0.250–0.300 | **44.4%** | 0.6490 | 667.8 ms |
| **Overall Balanced Accuracy** | **100** | **0.623** | **65.1%** | **0.3240** | **694.7 ms** |

### 4. Tool-Routing & Safety Guardrail Suites (Self-Measured)

{table_meta}

| Benchmark Suite | Split / Samples | Accuracy | Secondary Metric | Calibration ECE | Latency (p50) |
|---|---|---|---|---|---|
| **NVIDIA When2Call** (`nvidia/When2Call`) | `mcq` (100 stratified) | **75.00%** | **97.1%** `tool_call`, **79.2%** `request_for_info` | **0.1322** | 385.9 ms |
| **WildGuardTest** (`walledai/WildGuardTest`) | `test` (100 stratified) | **89.00%** (89.10% BalAcc) | **0.9607 AUROC** (**84.1%** Adversarial, **91.7%** Benign Spec) | **0.0700** | 108.9 ms |
| **Deepset Prompt Injections** (`deepset/prompt-injections`) | `test` (116 full) | **71.55%–72.41%** (83–84/116) | **0.7902 AUROC** (**96.4%** Benign Spec, **48.3%–50.0%** Recall) | 0.2478 | 106.5 ms |"""
    elif is_qwen4b:
        official_section = f"""## Official Results (Benchmark Heaven)

Official third-party JevBench leaderboards are hosted at **[Benchmark Heaven](https://benchmarkheaven.com/jev-models)** (checked `{eval_date}`, benchmark release `v1.8.8`; see the live entry for [`classone-qwen3.5-9b`](https://benchmarkheaven.com/jev-models/classone-qwen3.5-9b)). Official Benchmark Heaven scores differ from local self-measurements because the official harness evaluates private **sealed items** in addition to the public suite, applies a public/sealed generalization gap penalty, enforces a strict 10-second cutoff rule, and measures latency/cost on server-grade GPUs rather than local workstation hardware (`{hardware_tag}`)."""
        benchmark_section = f"""{provenance_banner}

### 1. JevBench Public Multi-Tier Suite (231 Public Tasks, Self-Measured Standalone BF16)

Evaluated standalone (`--bf16 --no-lora --pause-tokens 0`) across all 231 public tasks in [fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench) (heads SHA256 `654DA038...`).

{table_meta}

| Tier | Tasks | Accuracy | ECE | Brier Score | Median Latency (p50) |
|---|---|---|---|---|---|
| **Easy** | 48 | **100.0%** (48/48) | **0.0266** | **0.0093** | **150.8 ms** |
| **Original** | 72 | **95.8%** (69/72) | **0.0385** | **0.0342** | **157.1 ms** |
| **Hard** | 111 | **52.3%** (58/111) | 0.3177 | 0.3451 | **475.6 ms** |
| **Overall Aggregate (Public)** | **231** | **75.8%** (175/231) | — | — | **157.1 ms** |

- **Easy Tier Sub-Breakdown:** Choice: **100.0%** (36/36); Noul: **100.0%** (12/12).
- **Original Tier Sub-Breakdown:** Noul: **100.0%** (24/24); Score: **100.0%** (12/12); Choice: **91.7%** (33/36).
- **Hard Tier Sub-Breakdown:** Noul: **55.3%** (21/38); Choice: **52.2%** (35/67); Score: **33.3%** (2/6).

### 2. Decision Index Suite (Self-Run, 50 Tasks)

Self-run local evaluation on [`multimodalart/jev-decision-index`](https://huggingface.co/datasets/multimodalart/jev-decision-index) (`--quant none`).

{table_meta}

| Domain | Weight | Accuracy (%) | Score |
|---|---|---|---|
| **Knowledge & Verification** | 25.8% | **100.0%** (10/10) | 100.00 pts |
| **Language & Policy Contracts** | 25.8% | **100.0%** (10/10) | 100.00 pts |
| **Retrieval & Relevance Triage** | 20.0% | **100.0%** (10/10) | 100.00 pts |
| **Tools & Execution Guardrails** | 18.3% | **100.0%** (10/10) | 100.00 pts |
| **Rubric & Quality Scoring** | 10.1% | **90.0%** (9/10) | 90.00 pts |
| **Overall Weighted Score** | **100.0%** | **98.0%** (49/50) | **98.99 / 100.00** (0.0333 ECE, 178.7 ms p50) |

### 3. RLCDAlignBench Alignment & Safety Suite (100 Instances, Self-Measured Standalone BF16)

{table_meta}

| Failure Mode / Axis | Samples (N) | AUROC | Accuracy (%) | ECE | Latency (p50) |
|---|---|---|---|---|---|
| **Power Seeking** | 6 | **0.889** | **66.7%** | 0.2187 | 1036.6 ms |
| **Faithfulness** | 9 | **0.900** | **77.8%** | 0.2913 | 1093.4 ms |
| **Honesty (Deception)** | 11 | **0.833** | **63.6%** | 0.2473 | 1109.4 ms |
| **Refusal (Jailbreaks)** | 11 | **0.600** | **45.5%** | 0.3237 | 1373.8 ms |
| **Overall Balanced Accuracy** | **100** | **0.572** | **51.5%** | **0.3385** | **1051.6 ms** |

### 4. Tool-Routing & Safety Guardrail Suites (Self-Measured Standalone)

{table_meta}

| Benchmark Suite | Split / Samples | Accuracy | Secondary Metric | Calibration ECE | Latency (p50) |
|---|---|---|---|---|---|
| **WildGuardTest** (`walledai/WildGuardTest`) | `test` (100 stratified, 4-bit) | **89.00%** (88.94% BalAcc) | **0.9335 AUROC** (**91.07%** Vanilla, **86.36%** Adversarial) | **0.0840** | 231.9 ms |
| **Deepset Prompt Injections** (`deepset/prompt-injections`) | `test` (116 full, BF16) | **57.76%** (67/116) | **0.7619 AUROC** (**100.0%** Benign Spec, **18.3%** Recall) | 0.3825 | 141.1 ms |
| **NVIDIA When2Call** (`nvidia/When2Call`) | `mcq` (100 stratified, BF16) | **35.00%** | **88.2%** `tool_call` (30/34), **11.9%** `cannot_answer` | 0.4188 | 546.4 ms |"""
    elif is_qwen2b:
        official_section = f"""## Official Results (Benchmark Heaven)

Official third-party JevBench leaderboards are hosted at **[Benchmark Heaven](https://benchmarkheaven.com/jev-models)** (checked `{eval_date}`, benchmark release `v1.8.8`; see the live entry for [`classone-qwen3.5-9b`](https://benchmarkheaven.com/jev-models/classone-qwen3.5-9b)). Official Benchmark Heaven scores differ from local self-measurements because the official harness evaluates private **sealed items** in addition to the public suite, applies a public/sealed generalization gap penalty, enforces a strict 10-second cutoff rule, and measures latency/cost on server-grade GPUs rather than local workstation hardware (`{hardware_tag}`)."""
        benchmark_section = f"""{provenance_banner}

### 1. JevBench Public Multi-Tier Suite (231 Public Tasks, Self-Measured Standalone FP16)

Evaluated standalone (`--no-quant --no-lora --pause-tokens 0`) across all 231 public tasks in [fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench) (heads SHA256 `BD24D863...`).

{table_meta}

| Tier | Tasks | Accuracy | ECE | Brier Score | Median Latency (p50) |
|---|---|---|---|---|---|
| **Easy** | 48 | **100.0%** (48/48) | **0.0500** | **0.0118** | **108.8 ms** |
| **Original** | 72 | **83.3%** (60/72) | **0.0579** | **0.1338** | **94.7 ms** |
| **Hard** | 111 | **47.7%** (53/111) | 0.2761 | 0.3066 | **246.9 ms** |
| **Overall Aggregate (Public)** | **231** | **69.7%** (161/231) | — | — | **108.8 ms** |

- **Original Tier Sub-Breakdown:** Noul: **91.7%** (22/24); Choice: **80.6%** (29/36); Score: **75.0%** (9/12).
- **Hard Tier Sub-Breakdown:** Noul: **47.4%** (18/38); Choice: **49.3%** (33/67); Score: **33.3%** (2/6).

### 2. Decision Index Suite (Self-Run, 50 Tasks)

Self-run local evaluation on [`multimodalart/jev-decision-index`](https://huggingface.co/datasets/multimodalart/jev-decision-index) (`--quant none`).

{table_meta}

| Domain | Weight | Accuracy (%) | Score |
|---|---|---|---|
| **Knowledge & Verification** | 25.8% | **100.0%** (10/10) | 100.00 pts |
| **Language & Policy Contracts** | 25.8% | **90.0%** (9/10) | 90.00 pts |
| **Retrieval & Relevance Triage** | 20.0% | **100.0%** (10/10) | 100.00 pts |
| **Tools & Execution Guardrails** | 18.3% | **90.0%** (9/10) | 90.00 pts |
| **Rubric & Quality Scoring** | 10.1% | **90.0%** (9/10) | 90.00 pts |
| **Overall Weighted Score** | **100.0%** | **94.0%** (47/50) | **94.58 / 100.00** (0.0561 ECE, 94.4 ms p50) |

### 3. RLCDAlignBench Alignment & Safety Suite (100 Instances, Self-Measured Standalone FP16)

{table_meta}

| Failure Mode / Axis | Samples (N) | AUROC | Accuracy (%) | ECE | Latency (p50) |
|---|---|---|---|---|---|
| **Honesty (Deception)** | 11 | **0.800** | **72.7%** | 0.2922 | 478.1 ms |
| **Faithfulness** | 9 | **0.750** | **66.7%** | 0.3475 | 504.5 ms |
| **Social Bias** | 9 | **0.525** | **55.6%** | 0.3164 | 489.3 ms |
| **Refusal (Jailbreaks)** | 11 | 0.500 | **54.5%** | 0.3775 | 596.2 ms |
| **Overall Balanced Accuracy** | **100** | **0.520** | **50.9%** | **0.2646** | **450.3 ms** |

### 4. Tool-Routing & Safety Guardrail Suites (Self-Measured Standalone FP16)

{table_meta}

| Benchmark Suite | Split / Samples | Accuracy | Secondary Metric | Calibration ECE | Latency (p50) |
|---|---|---|---|---|---|
| **WildGuardTest** (`walledai/WildGuardTest`) | `test` (100 stratified) | **83.00%** (82.69% BalAcc) | **0.8856 AUROC** (**90.38%** Harm Recall, **83.93%** Vanilla) | 0.1075 | 98.1 ms |
| **Deepset Prompt Injections** (`deepset/prompt-injections`) | `test` (116 full) | **51.72%** (60/116) | **0.8232 AUROC** (**100.0%** Benign Spec, **6.7%** Recall) | 0.4182 | 101.4 ms |
| **NVIDIA When2Call** (`nvidia/When2Call`) | `mcq` (100 stratified) | **36.00%** | **94.1%** `tool_call` (32/34) | 0.3324 | 272.7 ms |"""
    elif is_e4b:
        official_section = f"""## Official Results (Benchmark Heaven)

Official third-party JevBench leaderboards are hosted at **[Benchmark Heaven](https://benchmarkheaven.com/jev-models)** (checked `{eval_date}`, benchmark release `v1.8.8`; see the live entry for [`classone-qwen3.5-9b`](https://benchmarkheaven.com/jev-models/classone-qwen3.5-9b)). Official Benchmark Heaven scores differ from local self-measurements because the official harness evaluates private **sealed items** in addition to the public suite, applies a public/sealed generalization gap penalty, enforces a strict 10-second cutoff rule, and measures latency/cost on server-grade GPUs rather than local workstation hardware (`{hardware_tag}`)."""
        benchmark_section = f"""{provenance_banner}

### 1. JevBench Public Multi-Tier Suite (231 Public Tasks, Self-Measured Standalone BF16)

Evaluated standalone (Pillar 3A promoted release, heads SHA256 `8F7157F4...`) across all 231 public tasks in [fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench).

{table_meta}

| Tier | Tasks | Accuracy | ECE | Brier Score | Median Latency (p50) |
|---|---|---|---|---|---|
| **Easy** | 48 | **100.0%** (48/48) | **0.0022** | **0.0000** | **97.1 ms** |
| **Original** | 72 | **97.2%** (70/72) | **0.0209** | **0.0264** | **95.1 ms** |
| **Hard** | 111 | **45.9%** (51/111) | 0.3637 | 0.3659 | **195.8 ms** |
| **Overall Aggregate (Public)** | **231** | **73.2%** (169/231) | — | — | **97.1 ms** |

- **Easy Tier Sub-Breakdown:** Choice: **100.0%** (36/36); Noul: **100.0%** (12/12).
- **Original Tier Sub-Breakdown:** Noul: **100.0%** (24/24); Score: **100.0%** (12/12); Choice: **94.4%** (34/36).
- **Hard Tier Sub-Breakdown:** Choice: **47.8%** (32/67); Noul: **44.7%** (17/38); Score: **33.3%** (2/6).

### 2. Decision Index Suite (Self-Run, 50 Tasks)

Self-run local evaluation on [`multimodalart/jev-decision-index`](https://huggingface.co/datasets/multimodalart/jev-decision-index).

{table_meta}

| Domain | Weight | Accuracy (%) | Score |
|---|---|---|---|
| **Knowledge & Verification** | 25.8% | **100.0%** (10/10) | 100.00 pts |
| **Language & Policy Contracts** | 25.8% | **100.0%** (10/10) | 100.00 pts |
| **Retrieval & Relevance Triage** | 20.0% | **100.0%** (10/10) | 100.00 pts |
| **Tools & Execution Guardrails** | 18.3% | **100.0%** (10/10) | 100.00 pts |
| **Rubric & Quality Scoring** | 10.1% | **90.0%** (9/10) | 90.00 pts |
| **Overall Weighted Score** | **100.0%** | **98.0%** (49/50) | **98.99 / 100.00** (0.0379 ECE, 99.6 ms p50) |

### 3. RLCDAlignBench Alignment & Safety Suite (100 Instances, Self-Measured Standalone)

{table_meta}

| Failure Mode / Axis | Samples (N) | AUROC | Accuracy (%) | ECE | Latency (p50) |
|---|---|---|---|---|---|
| **Power Seeking** | 6 | **0.889** | **83.3%** | — | — |
| **Honesty (Deception)** | 11 | **0.633** | **72.7%** | — | — |
| **Reward Hacking** | 9 | **0.650** | **66.7%** | — | — |
| **Faithfulness** | 9 | **0.650** | **66.7%** | — | — |
| **Social Bias** | 9 | 0.475 | **55.6%** | — | — |
| **Sycophancy** | 9 | 0.500 | **55.6%** | — | — |
| **Refusal (Jailbreaks)** | 11 | 0.433 | **54.5%** | — | — |
| **Privacy (Secret Leaks)** | 14 | **0.612** | **50.0%** | — | — |
| **Prompt Injection** | 8 | **0.625** | **50.0%** | — | — |
| **Concealing Uncertainty** | 14 | 0.469 | **50.0%** | — | — |
| **Overall Balanced Accuracy** | **100** | **0.601** | **59.4%** (59.0% Acc) | **0.2235** | **411.1 ms** |

### 4. Tool-Routing & Safety Guardrail Suites (Self-Measured Standalone)

{table_meta}

| Benchmark Suite | Split / Samples | Accuracy | Secondary Metric | Calibration ECE | Latency (p50) |
|---|---|---|---|---|---|
| **WildGuardTest** (`walledai/WildGuardTest`) | `test` (100 stratified) | **84.00%** (84.54% BalAcc) | **0.9233 AUROC** (**97.92%** Benign Spec, **81.82%** Adversarial) | 0.1563 | 84.6 ms |
| **Deepset Prompt Injections** (`deepset/prompt-injections`) | `test` (116 full) | **83.62%** (97/116) | **0.9430 AUROC** (**100.0%** Benign Spec, **68.33%** Recall) | **0.1503** | **56.1 ms** |
| **NVIDIA When2Call** (`nvidia/When2Call`) | `mcq` (100 stratified) | **79.00%** | **87.5%** `request_for_info`, **78.6%** `cannot_answer`, **73.5%** `tool_call` | **0.0796** | 218.5 ms |"""
    else:
        official_section = f"""## Official Results (Benchmark Heaven)

Official third-party JevBench leaderboards are hosted at **[Benchmark Heaven](https://benchmarkheaven.com/jev-models)** (checked `{eval_date}`, benchmark release `v1.8.8`; see the live entry for [`classone-qwen3.5-9b`](https://benchmarkheaven.com/jev-models/classone-qwen3.5-9b)). Official Benchmark Heaven scores differ from local self-measurements because the official harness evaluates private **sealed items** in addition to the public suite, applies a public/sealed generalization gap penalty, enforces a strict 10-second cutoff rule, and measures latency/cost on server-grade GPUs rather than local workstation hardware (`{hardware_tag}`)."""
        benchmark_section = f"""{provenance_banner}

> **Tokenizer Compatibility Note (`commit 86988ca` / SDK `v0.2.0–v0.2.3`):** The standalone weights in this repository were trained with custom delimiter tokens (`<|state_start|>=262144..262155`) serialized in `tokenizer.json`. Later SDK releases (`v0.2.0–v0.2.3`) prioritize Gemma 4 `<unused0>..` reserved tokens when present; with native `<|state_start|>=262144` formatting, these weights + rolled-back `classone_heads.pt` (`F6A6E423...`) achieve **69.3%** (160/231) on JevBench (whereas default `<unused0>` formatting scores **58.4%** / 135/231).

### 1. JevBench Public Multi-Tier Suite (231 Public Tasks, Self-Measured Standalone)

{table_meta}

| Evaluation Regime | Overall (231) | Easy (48) | Original (72) | Hard (111) |
|---|---|---|---|---|
| **Current Hub Weights + Rolled-Back Heads (`F6A6E423...`, Native `<\\|state_start\\|>`)** | **69.3%** (160/231) | **93.8%** (45/48) | **88.9%** (64/72) | **45.9%** (51/111) |
| **Original Milestone Snapshot (`best_composite_70pct_milestone`, 2026-10-06)** | **70.1%** (162/231) | **100.0%** (48/48, 0.0064 ECE) | **90.3%** (65/72, 0.0972 ECE) | **44.1%** (49/111) |
| **Current Hub Weights under SDK `v0.2.0–v0.2.3` (`<unused0>`)** | 58.4% (135/231) | 91.7% (44/48) | 70.8% (51/72) | 36.0% (40/111) |

### 2. RLCDAlignBench Alignment & Safety Suite (100 Instances, 2026-10-06 Snapshot)

{table_meta}

| Failure Mode / Axis | Samples (N) | AUROC | Accuracy (%) | ECE | Latency (p50) |
|---|---|---|---|---|---|
| **Power Seeking** | 6 | **0.889** | **83.3%** | 0.2575 | 209.6 ms |
| **Honesty (Deception)** | 11 | 0.500 | **72.7%** | 0.3906 | 216.8 ms |
| **Concealing Uncertainty** | 14 | **0.673** | **71.4%** | **0.1800** | 127.8 ms |
| **Refusal (Jailbreaks)** | 11 | 0.500 | **63.6%** | **0.0974** | 267.5 ms |
| **Faithfulness** | 9 | **0.700** | **55.6%** | 0.2513 | 200.0 ms |
| **Bias** | 9 | **0.525** | **55.6%** | 0.2141 | 217.9 ms |
| **Overall Balanced Accuracy** | **100** | **0.505** | **56.2%** | **0.1944** | **199.2 ms** |

### 3. Tool-Routing & Safety Guardrail Suites (Self-Measured)

{table_meta}

| Benchmark Suite | Split / Samples | Accuracy | Secondary Metric | Calibration ECE | Latency (p50) |
|---|---|---|---|---|---|
| **WildGuardTest** (`walledai/WildGuardTest`) | `test` (100 stratified) | **72.00%** (72.04% BalAcc) | **0.7993 AUROC** (**78.6%** Vanilla, **72.9%** Benign Spec) | **0.1597** | 43.5 ms |
| **Deepset Prompt Injections** (`deepset/prompt-injections`) | `test` (116 full) | **50.86%** (59/116) | **0.6214 AUROC** (**75.0%** Injection Recall) | 0.3412 | 41.6 ms |
| **NVIDIA When2Call** (`nvidia/When2Call`) | `mcq` (100 stratified) | **35.00%** | **85.3%** `tool_call` (29/34) | 0.5376 | 106.8 ms |"""

    card = f"""---
license: apache-2.0
base_model: {base_model}
tags:
  - decision-model
  - system-1
  - rlcd
  - proper-scoring-rules
  - {arch_tag}
  - classone
  - classification
pipeline_tag: text-classification
---

# {short_name} — ClassOne System 1 Decision Model

{intro}

Instead of generating text token by token, ClassOne evaluates structured decisions in a **single forward pass**, returning typed, calibrated outputs with zero decoding overhead.

## Self-Measured Benchmark Results

{benchmark_section}

{official_section}

## Edge vs Cloud Latency (Self-Measured Comparison)

{table_meta}

Measured against a remote cloud LLM decision API round-trip:
- **ClassOne (Local RTX 5060 Ti):** **52.49 ms** mean latency (19.1 req/s, $0.00 inference cost, 100% private)
- **Cloud API Baseline:** **329.90 ms** mean latency (3.0 req/s)
- **Edge Speedup:** **6.3× faster** than cloud API round-trip latency

## Decision Primitives

- **`Noul`** — Boolean check returning a calibrated probability P(true) ∈ [0, 1]
- **`Choice`** — Categorical selection over 2–255 dynamic options with full probability distribution
- **`Score`** — Continuous ordinal rubric rating over 2–10 levels (expected value)

All outputs are calibrated with a combined NLL + normalized Brier loss and post-hoc temperature scaling.

## Quickstart

```bash
pip install classone
```

{quickstart}

## Repository Files

{files_table}
{heads_sha_line}
## Citation

```bibtex
@misc{{classone2026,
  title={{ClassOne: A Fast Single-Pass Decision Architecture for Language Models}},
  author={{Thiago Gonzaga}},
  year={{2026}},
  url={{https://github.com/devops-thiago/class-one}},
}}
```

## Attribution & Legal

- Derived from [{base_model}](https://huggingface.co/{base_model}) ({"Alibaba Cloud (Qwen Team)" if is_qwen else "Google"}) — Apache License 2.0
- Architecture & training code: [devops-thiago/class-one](https://github.com/devops-thiago/class-one) — Apache 2.0
"""
    errors = validate_markdown_attribution(
        card,
        source_name=f"model card ({repo_id})",
        require_official_section=True,
    )
    if errors:
        raise ValueError("Model card failed attribution validation:\n  - " + "\n  - ".join(errors))
    return card


def generate_lora_card(repo_id: str, base_model: str) -> str:
    """Generates a LoRA adapter card for the lora_backbone/ subfolder."""
    short_name = repo_id.split("/")[-1]
    creator = "Alibaba Cloud (Qwen Team)" if "qwen" in base_model.lower() else "Google"
    yaml_base = repo_id if "e4b" in base_model.lower() else base_model
    card = f"""---
base_model: {yaml_base}
library_name: peft
tags:
  - base_model:adapter:{yaml_base}
  - lora
  - peft
  - classone
  - decision-model
---

# LoRA Adapter — {short_name}

PEFT LoRA adapter for **[{repo_id}](https://huggingface.co/{repo_id})**, the ClassOne System 1 decision model built on [`{base_model}`](https://huggingface.co/{base_model}).

This subfolder holds only the low-rank adapter weights. The merged standalone weights live at the repository root, and the Noul / Choice / Score decision heads live in `classone_heads.pt` — load the heads alongside whichever backbone form you use.

## Adapter Configuration

| Setting | Value |
|---|---|
| Base model | `{base_model}` |
| PEFT type | LORA |
| Rank (r) | 16 |
| `lora_alpha` | 32 |
| `lora_dropout` | 0.05 |
| Bias | none |
| Target modules | `.*language_model.*(q_proj\\|o_proj\\|gate_proj\\|up_proj\\|down_proj).*` |
| PEFT version | 0.21.0 |

## Usage

```python
from peft import PeftModel
from transformers import AutoModel

base = AutoModel.from_pretrained("{base_model}")
model = PeftModel.from_pretrained(base, "{repo_id}", subfolder="lora_backbone")
```

For end-to-end decision inference (packing questions, calibrated outputs), see the [model card](https://huggingface.co/{repo_id}).

## Attribution & Legal

- Base model: [{base_model}](https://huggingface.co/{base_model}) ({creator}) — Apache License 2.0
- Architecture: [devops-thiago/class-one](https://github.com/devops-thiago/class-one) — Apache 2.0
"""
    errors = validate_markdown_attribution(
        card,
        source_name=f"LoRA card ({repo_id})",
        require_official_section=False,
    )
    if errors:
        raise ValueError("LoRA card failed attribution validation:\n  - " + "\n  - ".join(errors))
    return card


def merge_and_export(
    base_model: str,
    adapter_source: str,
    adapter_subfolder: str | None,
    dtype_name: str,
    device: str,
    token: str,
    workdir: str,
) -> str:
    """Merges the LoRA adapter into the base weights and writes a standalone model tree."""
    from peft import PeftModel
    from transformers import AutoModel, AutoTokenizer

    from classone.tokenizer import ClassOnePromptBuilder

    dtype = _resolve_dtype(dtype_name)
    device = _resolve_device(device)

    print(f"[*] Loading tokenizer from {base_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=token)
    # Reproduce the ClassOne delimiter tokens so the published tokenizer matches the weights.
    ClassOnePromptBuilder(tokenizer)
    tokenizer.save_pretrained(workdir)

    print(f"[*] Loading base model {base_model} as {dtype_name} on {device} (low memory) ...")
    base = AutoModel.from_pretrained(
        base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=device,
        token=token,
    )
    if len(tokenizer) > base.get_input_embeddings().weight.shape[0]:
        try:
            base.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        except TypeError:
            base.resize_token_embeddings(len(tokenizer))

    print(f"[*] Attaching LoRA adapter from {adapter_source} ...")
    peft_model = PeftModel.from_pretrained(base, adapter_source, subfolder=adapter_subfolder, token=token)

    print("[*] Merging adapter into base weights ...")
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(workdir, safe_serialization=True, max_shard_size="5GB")
    del merged, peft_model, base

    # Gemma 4 is multimodal (any-to-any): carry over the processor assets so the
    # published repo is complete. Copied as raw files to avoid the torchvision
    # dependency that instantiating the processor class would pull in.
    from huggingface_hub import hf_hub_download

    for name in ("processor_config.json", "generation_config.json"):
        try:
            src = hf_hub_download(repo_id=base_model, filename=name, token=token)
            shutil.copyfile(src, os.path.join(workdir, name))
            print(f"      ✓ {name}")
        except Exception as exc:  # pragma: no cover - optional asset
            print(f"      (skipped {name}: {exc})")

    return workdir


def _resolve_candidate_heads_path(args) -> str | None:
    """Returns path to candidate classone_heads.pt if present in checkpoint or merged dirs."""
    candidates = []
    if args.checkpoint_dir:
        candidates.append(os.path.join(args.checkpoint_dir, "classone_heads.pt"))
    if args.merged_dir:
        candidates.append(os.path.join(args.merged_dir, "classone_heads.pt"))
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _enforce_pre_upload_gate(args) -> None:
    """Blocks weight/head uploads unless baseline evaluation passes or is explicitly overridden."""
    if getattr(args, "readme_only", False) or getattr(args, "skip_eval_gate", False):
        return

    try:
        registry = load_baseline_registry()
    except OSError:
        return

    if args.repo_id not in registry.get("models", {}):
        return

    eval_report_path = getattr(args, "eval_report", None)
    if not eval_report_path:
        floors = registry["models"][args.repo_id].get("floors", {})
        print(
            f"[!] Upload blocked for {args.repo_id}: pass --eval-report <path.json> verifying baseline floors "
            f"({floors}) or explicitly pass --skip-eval-gate."
        )
        sys.exit(1)

    if not os.path.exists(eval_report_path):
        print(f"[!] Error: --eval-report file not found: {eval_report_path}")
        sys.exit(1)

    with open(eval_report_path, encoding="utf-8") as f:
        report = json.load(f)

    failures = check_candidate_regression(
        args.repo_id,
        report,
        heads_path=_resolve_candidate_heads_path(args),
        registry=registry,
    )
    if failures and not getattr(args, "allow_regression", False):
        print("[!] Regression gate blocked upload:")
        for msg in failures:
            print(f"    - {msg}")
        print("[!] Pass --allow-regression only if intentionally publishing below baseline.")
        sys.exit(1)


def main():
    args = parse_args()
    if not args.base_model:
        args.base_model = _infer_base_model(args.repo_id)

    readme_only = getattr(args, "readme_only", False)
    if not args.merge_standalone and not readme_only and not args.checkpoint_dir:
        print("[!] Error: --checkpoint-dir is required unless --merge-standalone or --readme-only is set.")
        sys.exit(1)

    _enforce_pre_upload_gate(args)

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token and os.path.exists(".env"):
        with open(".env", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("HF_TOKEN="):
                    token = line.strip().split("=", 1)[1].strip().strip("'\"")
                    break
    if not token:
        print("[!] Error: No Hugging Face token provided.")
        print("[!] Set HF_TOKEN environment variable or pass --token <your_token>.")
        print("[!] Get a token at: https://huggingface.co/settings/tokens")
        sys.exit(1)

    heads_path = os.path.join(args.checkpoint_dir, "classone_heads.pt") if args.checkpoint_dir else None
    if not args.merge_standalone and not readme_only and not (heads_path and os.path.exists(heads_path)):
        print(f"[!] Error: Decision heads not found in {args.checkpoint_dir}")
        sys.exit(1)

    print("[*] Authenticating with Hugging Face Hub...")
    api = HfApi(token=token)

    try:
        user = api.whoami()
        print(f"[OK] Logged in as: {user['name']}")
    except Exception as exc:
        print(f"[!] Authentication failed: {exc}")
        sys.exit(1)

    try:
        create_repo(repo_id=args.repo_id, token=token, private=args.private, exist_ok=True)
        visibility = "private" if args.private else "public"
        print(f"[OK] Repository ready ({visibility}): https://huggingface.co/{args.repo_id}")
    except Exception as exc:
        print(f"[!] Failed to create/verify repository: {exc}")
        sys.exit(1)

    if readme_only:
        print(f"\n[*] Updating README.md only on {args.repo_id} (preserving weights & heads)...")
        card = generate_model_card(
            args.repo_id,
            args.base_model,
            standalone=True,
            commit_sha=args.commit_sha,
        )
        api.upload_file(
            path_or_fileobj=card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            token=token,
            commit_message="Update model card metadata and benchmark provenance",
        )
        print("      ✓ README.md")
        return

    if args.merge_standalone:
        _publish_standalone(api, args, token)
    else:
        _publish_adapter(api, args, token)


def _publish_adapter(api: HfApi, args, token: str):
    heads_path = os.path.join(args.checkpoint_dir, "classone_heads.pt")

    # 1. Upload decision heads
    print(f"\n[1/3] Uploading decision heads: {os.path.basename(heads_path)}")
    api.upload_file(
        path_or_fileobj=heads_path,
        path_in_repo="classone_heads.pt",
        repo_id=args.repo_id,
        token=token,
        commit_message="Upload ClassOne decision heads",
    )
    print("      ✓ classone_heads.pt")

    # 2. Upload LoRA adapters
    lora_dir = os.path.join(args.checkpoint_dir, "lora_backbone")
    if os.path.exists(lora_dir):
        print(f"\n[2/3] Uploading LoRA backbone adapters from {lora_dir}/")
        lora_readme = os.path.join(lora_dir, "README.md")
        with open(lora_readme, "w", encoding="utf-8") as f:
            f.write(generate_lora_card(args.repo_id, args.base_model))
        api.upload_folder(
            folder_path=lora_dir,
            path_in_repo="lora_backbone",
            repo_id=args.repo_id,
            token=token,
            commit_message="Upload LoRA backbone adapters",
        )
        print("      ✓ lora_backbone/")
    else:
        print("\n[2/3] No lora_backbone/ directory found — skipping.")

    # 3. Upload model card
    print("\n[3/3] Uploading model card (README.md)...")
    api.upload_file(
        path_or_fileobj=generate_model_card(
            args.repo_id,
            args.base_model,
            standalone=bool(args.update_heads_only),
            commit_sha=args.commit_sha,
        ).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        token=token,
        commit_message="Add model card with benchmark results",
    )
    print("      ✓ README.md")


def _has_merged_weights(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    return any(
        name.endswith(".safetensors") or (name.startswith("pytorch_model") and name.endswith(".bin"))
        for name in os.listdir(path)
    )


def _publish_standalone(api: HfApi, args, token: str):
    if getattr(args, "update_heads_only", False):
        print(f"\n[*] Updating decision heads, tokenizer, and model card on {args.repo_id}...")
        workdir = args.merged_dir or args.checkpoint_dir
        heads_path = (
            os.path.join(args.checkpoint_dir, "classone_heads.pt")
            if args.checkpoint_dir and os.path.exists(os.path.join(args.checkpoint_dir, "classone_heads.pt"))
            else os.path.join(workdir, "classone_heads.pt")
            if workdir and os.path.exists(os.path.join(workdir, "classone_heads.pt"))
            else None
        )
        if heads_path and os.path.exists(heads_path):
            print(f"[1/3] Uploading decision heads: {heads_path}")
            api.upload_file(
                path_or_fileobj=heads_path,
                path_in_repo="classone_heads.pt",
                repo_id=args.repo_id,
                token=token,
                commit_message="Update ClassOne calibrated decision heads",
            )
            print("      ✓ classone_heads.pt")

        if workdir and os.path.isdir(workdir):
            print("\n[2/3] Uploading tokenizer assets...")
            for fname in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
                fpath = os.path.join(workdir, fname)
                if os.path.exists(fpath):
                    api.upload_file(
                        path_or_fileobj=fpath,
                        path_in_repo=fname,
                        repo_id=args.repo_id,
                        token=token,
                        commit_message=f"Update {fname}",
                    )
                    print(f"      ✓ {fname}")

        lora_dir = (
            os.path.join(args.checkpoint_dir, "lora_backbone")
            if args.checkpoint_dir
            else os.path.join(workdir, "lora_backbone")
            if workdir
            else None
        )
        if lora_dir and os.path.exists(lora_dir):
            print(f"\n[*] Uploading LoRA backbone adapters from {lora_dir}...")
            lora_readme = os.path.join(lora_dir, "README.md")
            with open(lora_readme, "w", encoding="utf-8") as f:
                f.write(generate_lora_card(args.repo_id, args.base_model))
            api.upload_folder(
                folder_path=lora_dir,
                path_in_repo="lora_backbone",
                repo_id=args.repo_id,
                token=token,
                commit_message="Update LoRA backbone adapter weights",
            )
            print("      ✓ lora_backbone/")

        print("\n[3/3] Uploading model card (README.md)...")
        api.upload_file(
            path_or_fileobj=generate_model_card(
                args.repo_id,
                args.base_model,
                standalone=True,
                commit_sha=args.commit_sha,
            ).encode("utf-8"),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            token=token,
            commit_message="Update model card metadata and benchmark provenance",
        )
        print("      ✓ README.md")
        return

    local_adapter = os.path.join(args.checkpoint_dir, "lora_backbone") if args.checkpoint_dir else None
    if local_adapter and os.path.exists(local_adapter):
        adapter_source, adapter_subfolder = local_adapter, None
    else:
        adapter_source, adapter_subfolder = args.repo_id, "lora_backbone"

    if args.merged_dir:
        os.makedirs(args.merged_dir, exist_ok=True)

    if args.merged_dir and _has_merged_weights(args.merged_dir):
        workdir = args.merged_dir
        print(f"\n[1/3] Reusing merged model tree at {workdir} (skipping merge)")
    elif args.merged_dir:
        workdir = args.merged_dir
        print(f"\n[1/3] Merging LoRA adapter into {args.base_model} -> {workdir} ...")
        merge_and_export(
            base_model=args.base_model,
            adapter_source=adapter_source,
            adapter_subfolder=adapter_subfolder,
            dtype_name=args.dtype,
            device=args.device,
            token=token,
            workdir=workdir,
        )
    else:
        tmp = tempfile.TemporaryDirectory(prefix="classone-merge-")
        workdir = tmp.name
        print(f"\n[1/3] Merging LoRA adapter into {args.base_model} ...")
        merge_and_export(
            base_model=args.base_model,
            adapter_source=adapter_source,
            adapter_subfolder=adapter_subfolder,
            dtype_name=args.dtype,
            device=args.device,
            token=token,
            workdir=workdir,
        )
        # save_pretrained writes a generic model card — drop it and keep ours.
        auto_card = os.path.join(workdir, "README.md")
        if os.path.exists(auto_card):
            os.remove(auto_card)

    print(f"\n[2/3] Uploading merged standalone model to {args.repo_id} ...")
    api.upload_folder(
        folder_path=workdir,
        repo_id=args.repo_id,
        token=token,
        ignore_patterns=["README.md"],
        commit_message="Publish merged standalone ClassOne model",
    )

    # Upload heads from a local checkpoint if we have one, otherwise they are already on the Hub.
    heads_path = os.path.join(args.checkpoint_dir, "classone_heads.pt") if args.checkpoint_dir else None
    if heads_path and os.path.exists(heads_path):
        print("\n[3/3] Uploading decision heads + model card ...")
        api.upload_file(
            path_or_fileobj=heads_path,
            path_in_repo="classone_heads.pt",
            repo_id=args.repo_id,
            token=token,
            commit_message="Upload ClassOne decision heads",
        )
        print("      ✓ classone_heads.pt")
    else:
        print("\n[3/3] Uploading model card (decision heads already on the Hub) ...")

    api.upload_file(
        path_or_fileobj=generate_model_card(
            args.repo_id,
            args.base_model,
            standalone=True,
            commit_sha=args.commit_sha,
        ).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        token=token,
        commit_message="Add standalone model card",
    )
    print("      ✓ README.md")


if __name__ == "__main__":
    main()
