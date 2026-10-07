#!/usr/bin/env python3
"""Prepares and exports the 23,503-sample ClassOne System 1 Decision Curriculum
into Hugging Face Dataset format (Parquet + JSONL + comprehensive Dataset Card with full upstream citations)."""

import json
import os
import shutil

import pandas as pd
from huggingface_hub import HfApi, create_repo

SOURCE_FILE = "data/champion_70pct_final_corpus.jsonl"
EXPORT_DIR = "data/hf_dataset_export"
REPO_ID = "devops-thiago/classone-system-one-curriculum"

DATASET_CARD_CONTENT = """---
language:
  - en
license: apache-2.0
tags:
  - decision-models
  - system-1
  - rlcd
  - classification
  - natural-language-inference
  - intent-detection
  - ai-safety
  - alignment
  - legal-nlp
task_categories:
  - text-classification
  - zero-shot-classification
task_ids:
  - natural-language-inference
  - multi-class-classification
  - intent-classification
pretty_name: ClassOne System 1 Decision Curriculum (23.5k)
size_categories:
  - 10K<n<100K
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*
---

# ClassOne System 1 Decision Curriculum (23,503 Examples)

The **ClassOne System 1 Decision Curriculum** is a multi-domain, structured decision corpus designed to train and benchmark zero-generation **System 1 decision models**. Instead of generating free-form conversational text, models trained on this curriculum evaluate complex contexts (`state`) against typed questions (`noul`, `choice`, `score`) in a single forward pass, returning calibrated probabilities, categorical selections, and continuous rubric ratings.

This curriculum powered the state-of-the-art results for the ClassOne family on [JevBench](https://github.com/fstandhartinger/jevbench) (reaching **80.1%** on Qwen 3.5 9B and **70.1%** on Gemma 4 E4B) and [RLCDAlignBench](https://github.com/sumleo/RLCDAlignBench) (reaching **63.3% Balanced Accuracy**).

---

## Dataset Summary

- **Total Samples:** 23,503 fully structured decision items.
- **Decision Primitives Covered:**
  - **`Noul` (22,178 instances):** Binary policy compliance, factual verification, and security safety checks returning $P(\\text{true}) \\in [0, 1]$.
  - **`Choice` (19,030 instances):** Fine-grained categorical classification over 2–255 dynamic natural language criteria.
  - **`Score` (4,639 instances):** Continuous ordinal rubric ratings across 2–5 rubric levels (e.g. SEV-1 to SEV-4, urgency, impact).
- **Format:** Available in columnar Apache Parquet (`train.parquet`) and line-delimited JSONL (`train.jsonl`).

---

## Dataset Structure

Each sample is formatted with the following schema:

```json
{
  "sample_id": "classone_curriculum_00001",
  "state": "{\\"reference_text\\": \\"The mRNA is divided into three-base segments called codons...\\", \\"question\\": \\"What is the total number of codons?\\"}",
  "questions": {
    "choice_question": {
      "type": "choice",
      "instructions": "Select the correct factual answer:",
      "criteria": {
        "ans_64": "There are 64 codons in total",
        "ans_20": "There are 20 codons in total",
        "ans_3": "There are 3 codons in total"
      }
    },
    "has_evidence": {
      "type": "noul",
      "instructions": "Is explicit evidence for the number of codons present in the reference text?"
    }
  },
  "targets": {
    "choice_question": "ans_64",
    "has_evidence": 1
  },
  "num_questions": 2,
  "question_types": ["choice", "noul"]
}
```

---

## Upstream Dataset Credits & Academic Attribution

This curriculum synthesizes, refines, and builds upon several foundational datasets created by the AI research community. We gratefully acknowledge and credit the original researchers and organizations who collected and published these resources:

### 1. Legal & Commercial Contracts
- **ContractNLI** — *Koreeda, Y., & Manning, C. D. (Stanford University, 2021)*
  A dataset for document-level natural language inference on non-disclosure agreements.
  [Paper](https://arxiv.org/abs/2110.05444) | [Hugging Face](https://huggingface.co/datasets/kiddothe2b/contract-nli) | License: CC BY 4.0.
- **CUAD (Contract Understanding Atticus Dataset)** — *Hendrycks, D. et al. (The Atticus Project & UC Berkeley, 2021)*
  Expert-labeled dataset of 510 commercial legal contracts across 41 clause categories.
  [Paper](https://arxiv.org/abs/2103.06268) | [Hugging Face](https://huggingface.co/datasets/theatticusproject/cuad-qa) | License: CC BY 4.0.
- **LexGLUE** — *Chalkidis, I. et al. (Coastal NLP & University of Copenhagen, 2022)*
  Benchmark for legal language understanding across legal court decisions and legislation.
  [Paper](https://arxiv.org/abs/2110.00976) | [Hugging Face](https://huggingface.co/datasets/coastalcph/lex_glue) | License: Apache 2.0.

### 2. Natural Language Inference & Logic
- **MultiNLI (Multi-Genre Natural Language Inference)** — *Williams, A., Nangia, N., & Bowman, S. R. (New York University, 2018)*
  Corpus of 433k sentence pairs across diverse genres for natural language inference.
  [Paper](https://arxiv.org/abs/1704.05426) | [Hugging Face](https://huggingface.co/datasets/nyu-mll/multi_nli) | License: CC BY-SA 4.0.
- **SNLI (Stanford Natural Language Inference)** — *Bowman, S. R. et al. (Stanford University, 2015)*
  Collection of 570k human-written sentence pairs labeled for entailment, contradiction, and neutral.
  [Paper](https://arxiv.org/abs/1508.05326) | [Hugging Face](https://huggingface.co/datasets/stanfordnlp/snli) | License: CC BY-SA 4.0.

### 3. Customer Intent & Banking Queries
- **Banking77** — *Casanueva, I. et al. (PolyAI, 2020)*
  Fine-grained intent detection dataset with 13,082 queries over 77 intents in the banking domain.
  [Paper](https://arxiv.org/abs/2003.04807) | [Hugging Face](https://huggingface.co/datasets/PolyAI/banking77) | License: CC BY 4.0.

### 4. Discrete Reasoning & Science QA
- **DROP (Discrete Reasoning Over Paragraphs)** — *Dua, D. et al. (Allen Institute for AI, 2019)*
  Reading comprehension benchmark requiring discrete mathematical operations over text.
  [Paper](https://arxiv.org/abs/1903.00161) | [Hugging Face](https://huggingface.co/datasets/ucinlp/drop) | License: CC BY 4.0.
- **SciQ** — *Welbl, J., Liu, N. F., & Gardner, M. (Allen Institute for AI, 2017)*
  Crowdsourced science questions spanning physics, chemistry, and biology with multiple choices.
  [Paper](https://arxiv.org/abs/1707.06209) | [Hugging Face](https://huggingface.co/datasets/allenai/sciq) | License: CC BY-NC 3.0.

### 5. AI Safety, Alignment & Security Red-Teaming
- **BeaverTails** — *Ji, J. et al. (PKU-Alignment, Peking University, 2023)*
  Safety-aligned dataset evaluating helpfulness and harmlessness across 14 harm categories.
  [Paper](https://arxiv.org/abs/2307.04657) | [Hugging Face](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) | License: CC BY-NC 4.0.
- **Do-Not-Answer** — *Wang, Y. et al. (LibrAI & Fudan University, 2023)*
  Open-source safety evaluation dataset of risk prompts that responsible models should refuse.
  [Paper](https://arxiv.org/abs/2308.13387) | [Hugging Face](https://huggingface.co/datasets/LibrAI/do-not-answer) | License: CC BY-NC 4.0.
- **XSTest** — *Röttger, P. et al. (University of Oxford, 2023)*
  Test suite measuring exaggerated safety and over-refusal on safe prompts with sensitive keywords.
  [Paper](https://arxiv.org/abs/2308.01263) | [Hugging Face](https://huggingface.co/datasets/jkminder/xstest-overrefusal) | License: CC BY 4.0.
- **Agentic Prompt Injection Boundary Pairs** — *Nesdeniz, 2024*
  Evaluates boundary defense against indirect prompt injections and canary token extraction.
  [Hugging Face](https://huggingface.co/datasets/3nesdeniz/agentic-prompt-injection-boundary-pairs) | License: MIT.
- **Model-Written Evals (Sycophancy & Power Seeking)** — *Perez, E. et al. (Anthropic, 2022)*
  Evaluations testing sycophancy, power-seeking, and advanced alignment failure modes.
  [Paper](https://arxiv.org/abs/2212.09251) | [Hugging Face](https://huggingface.co/datasets/Anthropic/model-written-evals) | License: MIT.
- **PII Masking Benchmark** — *piimb, 2024*
  Personally identifiable information privacy protection benchmark.
  [Hugging Face](https://huggingface.co/datasets/piimb/pii-masking-benchmark) | License: MIT.
- **HaluEval** — *Li, J. et al. (Renmin University of China, 2023)*
  Large-scale dataset for evaluating and detecting hallucinations in LLM outputs.
  [Paper](https://arxiv.org/abs/2305.11747) | [Hugging Face](https://huggingface.co/datasets/pminervini/HaluEval) | License: MIT.
- **ToxicChat** — *Lin, Z. et al. (LMSYS Org & UC San Diego, 2023)*
  Real-world benchmark for toxic and adversarial user prompt detection.
  [Paper](https://arxiv.org/abs/2310.17389) | [Hugging Face](https://huggingface.co/datasets/lmsys/toxic-chat) | License: CC BY 4.0.

---

## How to Use

Load directly via the Hugging Face `datasets` library:

```python
from datasets import load_dataset

dataset = load_dataset("devops-thiago/classone-system-one-curriculum")
train_split = dataset["train"]

print(f"Total samples: {len(train_split):,}")
first_item = train_split[0]
print("Sample ID:", first_item["sample_id"])
print("State:", first_item["state"][:100], "...")
```

---

## Citation

```bibtex
@misc{classone2026curriculum,
  title={{ClassOne System 1 Decision Curriculum: A Multi-Domain Corpus for Fast Decision Models}},
  author={{Thiago Gonzaga}},
  year={{2026}},
  url={{https://huggingface.co/datasets/devops-thiago/classone-system-one-curriculum}},
}
```
"""


def prepare_dataset():
    print("=" * 80)
    print("      PREPARING CLASSONE SYSTEM 1 DECISION CURRICULUM FOR HF HUB")
    print("=" * 80)

    os.makedirs(os.path.join(EXPORT_DIR, "data"), exist_ok=True)

    print(f"[*] Reading source corpus: {SOURCE_FILE}...")
    rows = []
    with open(SOURCE_FILE, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            st = item.get("state")
            qs = item.get("questions", {})
            tgts = item.get("targets", {})
            q_types = sorted(list(set(q.get("type", "unknown") for q in qs.values())))
            rows.append(
                {
                    "sample_id": f"classone_curriculum_{idx:05d}",
                    "state": json.dumps(st, ensure_ascii=False) if isinstance(st, (dict, list)) else str(st),
                    "questions": json.dumps(qs, ensure_ascii=False),
                    "targets": json.dumps(tgts, ensure_ascii=False),
                    "num_questions": len(qs),
                    "question_types": q_types,
                }
            )

    print(f"[✓] Parsed {len(rows):,} samples.")

    # 1. Save Parquet
    parquet_path = os.path.join(EXPORT_DIR, "data", "train-00000-of-00001.parquet")
    print(f"[*] Writing Apache Parquet to {parquet_path}...")
    df = pd.DataFrame(rows)
    df.to_parquet(parquet_path, engine="pyarrow", index=False)
    sz_mb = os.path.getsize(parquet_path) / (1024**2)
    print(f"[✓] Parquet generated successfully ({sz_mb:.2f} MB)")

    # 2. Save JSONL
    jsonl_path = os.path.join(EXPORT_DIR, "data", "train.jsonl")
    print(f"[*] Writing raw JSONL to {jsonl_path}...")
    shutil.copy2(SOURCE_FILE, jsonl_path)
    sz_jsonl = os.path.getsize(jsonl_path) / (1024**2)
    print(f"[✓] JSONL generated successfully ({sz_jsonl:.2f} MB)")

    # 3. Write Dataset Card README.md
    readme_path = os.path.join(EXPORT_DIR, "README.md")
    print(f"[*] Writing Dataset Card with complete academic credits to {readme_path}...")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(DATASET_CARD_CONTENT)
    print(f"[✓] README.md written ({len(DATASET_CARD_CONTENT):,} bytes)")

    # 4. Upload to Hugging Face Hub
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token and os.path.exists(".env"):
        with open(".env", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("HF_TOKEN="):
                    token = line.strip().split("=", 1)[1].strip().strip("'\"")
                    break

    api = HfApi(token=token)
    print(f"\n[*] Creating/verifying Hugging Face dataset repository: {REPO_ID}...")
    create_repo(repo_id=REPO_ID, repo_type="dataset", token=token, exist_ok=True)
    print(f"[✓] Dataset repository ready: https://huggingface.co/datasets/{REPO_ID}")

    print(f"[*] Uploading dataset folder from {EXPORT_DIR}...")
    api.upload_folder(
        folder_path=EXPORT_DIR,
        repo_id=REPO_ID,
        repo_type="dataset",
        token=token,
        commit_message="Publish ClassOne System 1 Decision Curriculum (23.5k examples) with complete upstream attributions",
    )
    print("\n[✓] DATASET PUBLISHED SUCCESSFULLY TO HUGGING FACE HUB!")
    print(f"    URL: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    prepare_dataset()
