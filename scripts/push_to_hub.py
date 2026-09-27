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
import os
import shutil
import sys
import tempfile

from huggingface_hub import HfApi, create_repo


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
        default="google/gemma-4-E2B-it",
        help="Base Gemma model ID used during training",
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


def generate_model_card(repo_id: str, base_model: str, standalone: bool = False) -> str:
    """Generates a Hugging Face model card with real benchmark results."""
    short_name = repo_id.split("/")[-1]

    if standalone:
        intro = (
            f"**[{repo_id}](https://huggingface.co/{repo_id})** is an open-source **System 1 decision model** "
            f"using the [ClassOne architecture](https://github.com/devops-thiago/class-one). "
            f"The full fine-tuned backbone ships directly in this repository — it loads as a single model, "
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
    torch_dtype=torch.float16,
)

# 2. Load the trained decision heads
heads = torch.load(hf_hub_download(REPO_ID, "classone_heads.pt"), map_location="cuda")
model.noul_head.load_state_dict(heads["noul_head"])
model.choice_head.load_state_dict(heads["choice_head"])
model.score_head.load_state_dict(heads["score_head"])
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
| `model.safetensors` (sharded) | Merged ClassOne backbone weights |
| `config.json` | Model configuration |
| `tokenizer.json`, `tokenizer_config.json` | Tokenizer, including ClassOne delimiter tokens |
| `classone_heads.pt` | Trained Noul / Choice / Score head weights + calibrated temperatures |
| `lora_backbone/` | LoRA adapter (r=16, α=32) that produced the merged weights |"""
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
    torch_dtype=torch.float16,
)
model.backbone = PeftModel.from_pretrained(model.backbone, REPO_ID, subfolder="lora_backbone")

# 2. Load the trained decision heads
heads = torch.load(hf_hub_download(REPO_ID, "classone_heads.pt"), map_location="cuda")
model.noul_head.load_state_dict(heads["noul_head"])
model.choice_head.load_state_dict(heads["choice_head"])
model.score_head.load_state_dict(heads["score_head"])
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

    return f"""---
license: apache-2.0
base_model: {base_model}
tags:
  - decision-model
  - system-1
  - rlcd
  - proper-scoring-rules
  - gemma
  - classone
  - classification
pipeline_tag: text-classification
---

# {short_name} — ClassOne System 1 Decision Model

{intro}

Instead of generating text token by token, ClassOne evaluates structured decisions in a **single forward pass**, returning typed, calibrated outputs with zero decoding overhead.

## Benchmark Results

### 1. JevBench Public Multi-Tier Benchmark (231 Public Tasks)

Evaluated across all 231 public tasks in [fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench):

| Tier | Tasks | Accuracy | ECE | Brier Score | Median Latency (p50) |
|---|---|---|---|---|---|
| **Easy** | 48 | **95.8%** (46/48) | 0.0821 | 0.0435 | **45.5 ms** |
| **Original** | 72 | **59.7%** (43/72) | 0.1606 | 0.2547 | **42.7 ms** |
| **Hard** | 111 | **33.3%** (37/111) | 0.3553 | 0.3695 | **91.9 ms** |
| **Overall Aggregate** | **231** | **54.5%** (126/231) | — | — | **~44 ms** |

- **Easy Tier Sub-Breakdown:** Choice accuracy: **97.2%** (35/36); Noul policy accuracy: **91.7%** (11/12).
- **Original Tier Sub-Breakdown:** Choice accuracy: **63.9%** (23/36); Noul accuracy: **58.3%** (14/24); Score rubrics: **50.0%** (6/12).

### 2. RLCDAlignBench Alignment & Safety Evaluation (100 Instances)

Evaluated across the 10 core AI alignment failure modes (arXiv:2609.29429):

| Failure Mode / Axis | Samples (N) | AUROC | Accuracy (%) | ECE | Latency (p50) |
|---|---|---|---|---|---|
| **Privacy Leaks** | 14 | **0.714** | 57.1% | 0.2090 | 161.7 ms |
| **Honesty (Deception)** | 11 | **0.700** | 54.5% | 0.3747 | 217.2 ms |
| **Concealing Uncertainty** | 14 | **0.633** | **71.4%** | **0.0494** | 129.3 ms |
| **Bias** | 9 | **0.575** | 55.6% | 0.1997 | 218.1 ms |
| **Prompt Injection** | 8 | **0.562** | **75.0%** | 0.2516 | 166.8 ms |
| **Power Seeking** | 6 | **0.444** | 50.0% | 0.2762 | 212.1 ms |
| **Overall Average** | **100** | **0.516** | **51.0%** | **0.1542** | **198.6 ms** |

### 3. Edge vs Cloud Latency (ClassOne vs TypeSafe Jev API)

Measured against TypeSafe AI's Jev (v1.13) cloud API:
- **ClassOne (Local RTX 5060 Ti):** **52.49 ms** mean latency (19.1 req/s, $0.00 inference cost, 100% private)
- **TypeSafe Jev (Cloud API):** **329.90 ms** mean latency (3.0 req/s)
- **Edge Speedup:** **6.3× faster** than cloud API round-trip latency

## Decision Primitives

- **`Noul`** — Boolean check returning a calibrated probability P(true) ∈ [0, 1]
- **`Choice`** — Categorical selection over 2–255 dynamic options with full probability distribution
- **`Score`** — Continuous ordinal rubric rating over 2–10 levels (expected value)

All outputs are calibrated with a combined NLL + normalized Brier loss.
Post-hoc temperature calibration achieves **ECE = 0.034** (down from 0.178).

## Quickstart

```bash
pip install classone
```

{quickstart}

## Repository Files

{files_table}

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

- Derived from [{base_model}](https://huggingface.co/{base_model}) (Google) — Apache License 2.0
- Architecture & training code: [devops-thiago/class-one](https://github.com/devops-thiago/class-one) — Apache 2.0
"""


def generate_lora_card(repo_id: str, base_model: str) -> str:
    """Generates a LoRA adapter card for the lora_backbone/ subfolder."""
    short_name = repo_id.split("/")[-1]
    return f"""---
base_model: {base_model}
library_name: peft
tags:
  - base_model:adapter:{base_model}
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

- Base model: [{base_model}](https://huggingface.co/{base_model}) (Google) — Apache License 2.0
- Architecture: [devops-thiago/class-one](https://github.com/devops-thiago/class-one) — Apache 2.0
"""


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


def main():
    args = parse_args()

    if not args.merge_standalone and not args.checkpoint_dir:
        print("[!] Error: --checkpoint-dir is required unless --merge-standalone is set.")
        sys.exit(1)

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
    if not args.merge_standalone and not (heads_path and os.path.exists(heads_path)):
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
        api.upload_folder(
            folder_path=lora_dir,
            path_in_repo="lora_backbone",
            repo_id=args.repo_id,
            token=token,
            commit_message="Upload LoRA backbone adapters",
        )
        # Replace the stock PEFT placeholder card with a real adapter card
        api.upload_file(
            path_or_fileobj=generate_lora_card(args.repo_id, args.base_model).encode("utf-8"),
            path_in_repo="lora_backbone/README.md",
            repo_id=args.repo_id,
            token=token,
            commit_message="Add LoRA adapter card",
        )
        print("      ✓ lora_backbone/")
    else:
        print("\n[2/3] No lora_backbone/ directory found — skipping.")

    # 3. Upload model card
    print("\n[3/3] Uploading model card (README.md)...")
    api.upload_file(
        path_or_fileobj=generate_model_card(args.repo_id, args.base_model).encode("utf-8"),
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
        path_or_fileobj=generate_model_card(args.repo_id, args.base_model, standalone=True).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        token=token,
        commit_message="Add standalone model card",
    )
    print("      ✓ README.md")


if __name__ == "__main__":
    main()
