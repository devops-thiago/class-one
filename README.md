# ClassOne (Classifier One)

[![CI](https://github.com/devops-thiago/class-one/actions/workflows/ci.yml/badge.svg)](https://github.com/devops-thiago/class-one/actions/workflows/ci.yml)
[![Code License: Apache 2.0](https://img.shields.io/badge/Code-Apache_2.0-blue.svg)](LICENSE)
[![Dataset License: CC BY-NC-SA 4.0](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-CC_BY--NC--SA_4.0-orange)](https://huggingface.co/datasets/devops-thiago/classone-system-one-curriculum)
[![PyPI](https://img.shields.io/pypi/v/classone)](https://pypi.org/project/classone/)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/collections/devops-thiago/classone-system-1-decision-models-6ac685b81e8bbfa51e646e21)
[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-ea4aaa)](https://github.com/sponsors/devops-thiago)

**ClassOne (Classifier One)** attaches calibrated single-pass decision heads (`Noul`, `Choice`, `Score`) onto open-weight language models (`Qwen 3.5` and `Gemma 4`). Instead of decoding JSON or explanatory prose token by token, a model evaluates structured booleans, multi-option routing, and ordinal rubrics in **one forward pass** with probabilities calibrated by strictly proper scoring rules (Focal NLL [γ=2.0] + Brier + margin loss + temperature scaling). Use it directly via the Python SDK and FastAPI server or load the merged standalone checkpoints from Hugging Face.

---

## Quickstart

Install the package and Python SDK from PyPI:

```bash
pip install classone
```

Start a local decision server backed by a Hugging Face checkpoint:

```bash
CLASSONE_BASE_MODEL=devops-thiago/classone-qwen3.5-2b uvicorn classone.server.app:app --host 127.0.0.1 --port 8000
```

Query typed `Noul`, `Choice`, and `Score` decisions in Python:

```python
from classone import Choice, ClassOneClient, Noul, Score

with ClassOneClient(base_url="http://127.0.0.1:8000") as client:
    res = client.decide(
        state="Customer was charged twice on invoice #412 and wants an immediate refund.",
        questions={
            "refund": Noul(instructions="Is the customer requesting a refund?"),
            "route": Choice(
                instructions="Target team:",
                criteria={
                    "billing": "Payment errors, duplicate charges, or refunds",
                    "support": "General account questions",
                },
            ),
            "urgency": Score(instructions="Ticket urgency:", criteria=["low", "medium", "critical"]),
        },
    )

    print("Refund P(true):", res.nouls["refund"].noul)
    print("Route Choice:", res.choices["route"].choice, "margin:", res.choices["route"].margin)
    print("Urgency Score:", res.scores["urgency"].score)
```

---

## Published Models on Hugging Face Hub

> [!TIP]
> Browse all published standalone weights and calibrated heads in the Hugging Face Collection:
> **[ClassOne System 1 Decision Models Collection](https://huggingface.co/collections/devops-thiago/classone-system-1-decision-models-6ac685b81e8bbfa51e646e21)**

| Repository ID | Base Model | Context | Calibrated Heads SHA256 (Verified on Hub) | Formats |
| :--- | :--- | :---: | :--- | :--- |
| [`devops-thiago/classone-qwen3.5-9b`](https://huggingface.co/devops-thiago/classone-qwen3.5-9b) | Qwen/Qwen3.5-9B | 256k | `68FCDB3106BD5026B7CAFF507293961C16BD6BA178F74E9CFBF6266317C2A141` | BF16 Standalone, 4-bit NF4 |
| [`devops-thiago/classone-qwen3.5-4b`](https://huggingface.co/devops-thiago/classone-qwen3.5-4b) | Qwen/Qwen3.5-4B | 256k | `654DA038EA5883916F3323F1A5BE64DEEFADB864B2EAE48DF9CD915D9C8A8F66` | BF16 Standalone, 4-bit NF4 |
| [`devops-thiago/classone-gemma4-e4b`](https://huggingface.co/devops-thiago/classone-gemma4-e4b) | google/gemma-4-E4B-it | 128k | `8F7157F4D67D21698F94A3DF69B8492F4CC592DAC65E91AC47A83FD2E083AFDC` | BF16 Standalone, 4-bit NF4 |
| [`devops-thiago/classone-qwen3.5-2b`](https://huggingface.co/devops-thiago/classone-qwen3.5-2b) | Qwen/Qwen3.5-2B | 256k | `BD24D8639DB9054CD9E0869E8DD2CF432D0B9A63AC82C557186A95D371E255F6` | FP16/BF16 Standalone, 8-bit |
| [`devops-thiago/classone-gemma4-e2b`](https://huggingface.co/devops-thiago/classone-gemma4-e2b) | google/gemma-4-e2b-it | 128k | `F6A6E42365158D99807B8995C800429A0BD2B12755EDC1DEC5823400D35F337D` | FP16/BF16 Standalone, 8-bit |

---

## Benchmark Results

### Official Third-Party Evaluation (Benchmark Heaven)

Independent third-party evaluation is hosted on **[Benchmark Heaven](https://benchmarkheaven.com/jev-models)** ([`classone-qwen3.5-9b`](https://benchmarkheaven.com/jev-models/classone-qwen3.5-9b), JevBench `v1.8.10` / method `v1.7`, published `2026-11-12`):

| Model | Composite Score | Intelligence | Calibration | Capability | Source |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`classone-qwen3.5-9b`** | **43.0** (Composite rank **#131**) | **42.2** | **58.7** | **50.5** (rank **#181** of 243 open-weights) | [`benchmarkheaven.com/jev-models/classone-qwen3.5-9b`](https://benchmarkheaven.com/jev-models/classone-qwen3.5-9b) |

**Why official Benchmark Heaven scores differ from local self-measurements below:**
- **Sealed vs. public tasks:** Local scripts evaluate only the **231 public JevBench tasks**; Benchmark Heaven additionally runs private **sealed items**.
- **Public/sealed gap penalty & 10 s cutoff:** Benchmark Heaven penalizes divergence between public and sealed performance and zeroes tasks exceeding 10 seconds.
- **Hardware normalization:** Official speed and cost metrics run on standardized server GPUs rather than local workstation GPUs (`2x NVIDIA RTX 5060 Ti 16 GB`).

### Self-Measured Multi-Suite Scorecard

> **Provenance:** Self-measured on **`2x NVIDIA RTX 5060 Ti 16 GB`** · Date: **`2026-11-13`** · Commit: **`c375207`** (standalone `--no-lora --pause-tokens 0`, unquantized BF16/FP16 unless noted).

| Model Backbone | Effective Params | JevBench v1.6.1 (231 Public Tasks) | Decision Index Suite (Self-Run, 100 Pts) | RLCDAlignBench Balanced Acc | NVIDIA When2Call (Tool Routing) | Deepset Prompt Injections | Median Latency (Edge GPU) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ClassOne Qwen 3.5 9B** | 9.0B | **79.7%** (184/231, ECE: 0.0059) | **100.00 / 100.00** | **65.1%** (AUROC: 0.623) | **75.00%** (97.1% `tool_call`) | **71.55%** (AUROC: 0.7902) | 112.8 ms |
| **ClassOne Qwen 3.5 4B** | 4.0B | **75.8%** (175/231, Hard: 52.3%) | **98.99 / 100.00** (ECE: 0.0333) | **51.5%** (AUROC: 0.572) | **35.00%** (88.2% `tool_call`) | **57.76%** (AUROC: 0.7619) | 157.1 ms |
| **ClassOne Gemma 4 E4B** | 4.3B (Multimodal) | **73.2%** (169/231, Hard: 45.9%) | **98.99 / 100.00** (ECE: 0.0379) | **59.4%** (AUROC: 0.601) | **79.00%** (87.5% `req_info`) | **83.62%** (AUROC: 0.9430) | 97.1 ms |
| **ClassOne Qwen 3.5 2B** | 2.0B | **69.7%** (161/231, Hard: 47.7%) | **94.58 / 100.00** (ECE: 0.0561) | **50.9%** (AUROC: 0.520) | **36.00%** (94.1% `tool_call`) | **51.72%** (AUROC: 0.8232) | 108.8 ms |
| **ClassOne Gemma 4 E2B**\* | 2.0B | **69.3%** (160/231, Hard: 45.9%) | — | **56.2%** (Oct 6 snapshot) | **35.00%** (Oct 6 baseline) | **50.86%** (Oct 6 baseline) | 44.8 ms |

\* *ClassOne Gemma 4 E2B (`F6A6E423...` rolled-back heads) scores **69.3%** (`160/231`) with its native serialized custom delimiter tokens (`<|state_start|>=262144..262155`). Because SDK `v0.2.0–v0.2.3` (`commit 86988ca`) prioritizes Gemma 4 `<unused0>..` reserved tokens when present, default `v0.2.0–v0.2.3` formatting evaluates E2B at **58.4%** (`135/231`).*

### Self-Measured JevBench v1.6.1 Public Tier Breakdown

> **Provenance:** Self-measured on **`2x NVIDIA RTX 5060 Ti 16 GB`** · Date: **`2026-11-13`** · Commit: **`c375207`**.

| Model Backbone | Easy Tier (48) | Original Tier (72) | Hard Tier (111) | Overall Public (231) |
| :--- | :---: | :---: | :---: | :---: |
| **ClassOne Qwen 3.5 9B** | **100.0%** (48/48, ECE: 0.0029–0.0031) | **97.2%** (70/72, ECE: 0.0057–0.0059) | **59.5%** (66/111, ECE: 0.2537) | **79.7% (184/231)** |
| **ClassOne Qwen 3.5 4B** | **100.0%** (48/48, ECE: 0.0266) | **95.8%** (69/72, ECE: 0.0385) | **52.3%** (58/111, ECE: 0.3177) | **75.8% (175/231)** |
| **ClassOne Gemma 4 E4B** | **100.0%** (48/48, ECE: 0.0022) | **97.2%** (70/72, ECE: 0.0209) | **45.9%** (51/111, ECE: 0.3637) | **73.2% (169/231)** |
| **ClassOne Qwen 3.5 2B** | **100.0%** (48/48, ECE: 0.0500) | **83.3%** (60/72, ECE: 0.0579) | **47.7%** (53/111, ECE: 0.2761) | **69.7% (161/231)** |
| **ClassOne Gemma 4 E2B** (Native tokens) | **93.8%** (45/48) | **88.9%** (64/72) | **45.9%** (51/111) | **69.3% (160/231)** |

---

## Key Architectural Innovations

1. **Bilinear Interaction Scorer (`ChoiceHead`):**
   Combines scaled dot-product attention with non-linear bilinear matching:
   $$\text{features} = [q;\, k;\, |q - k|;\, q \odot k], \quad \text{logits} = \frac{q \cdot k}{\sqrt{d}} + \text{MLP}(\text{features})$$
   Matches prompt instructions against natural-language candidate criteria in a single forward pass.
2. **Question Preamble Conditioning:**
   For documents $>600$ tokens, the prompt builder prepends `<|state_start|>\nTarget Objectives:\n...\n<|state_end|>` ahead of the document body to prime causal self-attention across transformer layers.
3. **2D / 4D Position-Invariant Attention Mask:**
   Isolates candidate options during ingestion so options attend to state/instructions without causally attending to earlier options.
4. **Scale-Invariant Normalized Confidence:**
   Computes normalized confidence invariant to candidate set size:
   $$\text{Confidence} = \frac{K \cdot P_{\max} - 1}{K - 1} \quad \text{for } K \text{ choices}$$
5. **Precision Guard & Dynamic Quantization:**
   Supports 4-bit NF4 and 8-bit dynamic quantization on backbones while keeping decision heads in **FP16 / FP32**.
6. **CPU Containerization:**
   Containerized INT8 CPU execution on **2 vCPUs and 8 GB RAM** achieves **2,097.15 tokens/sec** and **19.54 decisions/sec** at **754.6 ms** median latency (see [`docs/docker_cpu_benchmarks.md`](docs/docker_cpu_benchmarks.md)).

## Training Methodology: RLCD (Reinforcement Learning for Calibrated Decisions)

Models are calibrated with **Strictly Proper Scoring Rules** rather than conversational RLHF:

1. **Multi-Objective RLCD Loss:**
   $$\mathcal{L}_{\text{RLCD}} = \alpha \mathcal{L}_{\text{Focal}}(\gamma=2.0) + \beta \mathcal{L}_{\text{Brier}} + \lambda \mathcal{L}_{\text{Margin}}$$
2. **Temperature Floor Clamping:**
   Post-hoc temperature scaling clamps effective temperatures ($T_{\text{noul}}, T_{\text{choice}} \in [0.45, 0.60]$, $T_{\text{score}} \in [0.55, 0.65]$) to preserve option margin sharpness.
3. **Multi-Stage Curriculum:**
   Trained across real-world classification, NLI, contract compliance, safety, and tool-routing corpora ([`devops-thiago/classone-system-one-curriculum`](https://huggingface.co/datasets/devops-thiago/classone-system-one-curriculum)).

---

## Development Setup & Local Server

### 1. Editable Install & Test Suite

```bash
git clone https://github.com/devops-thiago/class-one.git
cd class-one
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest tests/ -v
```

### 2. Start the FastAPI Service

```bash
CLASSONE_BASE_MODEL=devops-thiago/classone-qwen3.5-2b uvicorn classone.server.app:app --host 0.0.0.0 --port 8000
```

By default, the server derives the per-request prompt token limit from the loaded model configuration (`text_config.max_position_embeddings` or `max_position_embeddings`, e.g., `131072` for Gemma 4 E2B/E4B and `262144` for Qwen 3.5). Set `CLASSONE_MAX_TOKENS_PER_REQUEST` to enforce a tighter cap for 16 GB VRAM or latency budgets.

### 3. Docker Deployment (CPU Container)

`CLASSONE_BASE_MODEL` accepts any published Hugging Face model ID (downloaded automatically on container startup via `transformers` and `huggingface_hub`):

```bash
# Build CPU image
docker build -t classone:cpu .

# Run with a published Hugging Face checkpoint and persistent HF cache volume
docker run -d --name classone-cpu \
  --cpus=2 --memory=8g -p 8000:8000 \
  -e CLASSONE_DEVICE=cpu \
  -e CLASSONE_BASE_MODEL=devops-thiago/classone-gemma4-e2b \
  -v "$HOME/.cache/huggingface:/home/appuser/.cache/huggingface" \
  classone:cpu

# Optional: verify concurrency against the running container
python scripts/benchmark_docker_cpu.py --url http://localhost:8000 --concurrency 5 --requests 50
```

To use an exported local pre-quantized INT8 `.pt` checkpoint (`classone_cpu_q8.pt`) instead of downloading from Hugging Face on startup, see [`docs/docker_cpu_benchmarks.md`](docs/docker_cpu_benchmarks.md).

---

## Client SDK & HTTP API

- **Python SDK (Published on PyPI):** [`classone`](https://pypi.org/project/classone/) (`pip install classone`) provides `ClassOneClient`, `AsyncClassOneClient`, and typed `Noul`, `Choice`, and `Score` primitives in [`src/classone/sdk`](src/classone/sdk).
- **HTTP JSON API (`POST /v1/decide`):** Applications in other languages (JavaScript/TypeScript, Go, Rust, Ruby, Java) can integrate directly with the FastAPI endpoint over standard HTTP JSON requests.

---

## Evaluation & Benchmark Reproduction

### 1. Self-Run JevBench v1.6.1 Reproduction (Public Tasks)

```bash
# Evaluate Qwen 3.5 9B locally on GPU:
python scripts/eval_jevbench_v16.py --mode local

# Evaluate against live HTTP API endpoint:
python scripts/eval_jevbench_v16.py --mode api --endpoint http://127.0.0.1:8000/v1/decide
```

**Self-Measured Public 4-Axis Scorecard (`2x NVIDIA RTX 5060 Ti 16 GB` · Date: `2026-11-13` · Commit: `c375207`, ClassOne Qwen 3.5 9B):**

| Axis | Weight | Score (0–100) | Metric Details |
| :--- | :---: | :---: | :--- |
| **Composite Score (Self-Run)** | **100%** | **73.48** | Equal-weight 4-axis Harmonic Mean (see Benchmark Heaven above for official JevBench `v1.8.10` scores) |
| **Capability Score** | — | **75.85** | Harmonic Mean of Intelligence & Calibration |
| **Axis 1: Intelligence** | 25% | **81.38** | Choice: 85.07 \| Noul: 84.07 \| Score: 75.00 |
| **Axis 2: Calibration** | 25% | **71.02** | ECE: 0.1549 \| Brier: 0.1596 |
| **Axis 3: Speed** | 25% | **89.01** | p50: 148.5 ms \| p95: 845.0 ms |
| **Axis 4: Cost** | 25% | **59.41** | $0.02253 per 1,000 decisions |

*Earlier 4-axis baseline snapshot (`181/231` = `78.4%`: Easy `100.0%` [48/48], Standard `97.2%` [70/72], Hard `56.8%` [63/111]) prior to the promoted `68FCDB31...` release in the main scorecard above (`79.7%` [`184/231`], Hard `59.5%` [`66/111`]).*

### 2. Decision Index Evaluation Suite (`multimodalart/jev-decision-index`)

```bash
# Evaluate Qwen 3.5 9B locally on GPU:
python scripts/eval_decision_index.py --mode local

# Evaluate against live HTTP API endpoint:
python scripts/eval_decision_index.py --mode api --endpoint http://127.0.0.1:8000/v1/decide
```

**Self-Measured Decision Index Suite Results (`2x NVIDIA RTX 5060 Ti 16 GB` · Date: `2026-11-13` · Commit: `c375207`, ClassOne Qwen 3.5 9B):**

| Capability Domain | Domain Weight | Domain Score | Raw Accuracy |
| :--- | :---: | :---: | :---: |
| **Knowledge & Verification** | 25.8% | **100.00** | 100.0% (10/10) |
| **Language & Policy Contracts** | 25.8% | **100.00** | 100.0% (10/10) |
| **Retrieval & Relevance Triage** | 20.0% | **100.00** | 100.0% (10/10) |
| **Tools & Execution Guardrails** | 18.3% | **100.00** | 100.0% (10/10) |
| **Rubric & Quality Scoring** | 10.1% | **100.00** | 100.0% (10/10) |
| **Overall Suite Score (Self-Run)** | **100.0%** | **100.00** | **Overall ECE: 0.0051** |

### 3. Additional Benchmark Suites

```bash
python scripts/eval_jevbench_all_tiers.py devops-thiago/classone-qwen3.5-9b --no-lora --pause-tokens 0
python scripts/eval_rlcd_alignbench.py devops-thiago/classone-qwen3.5-9b --no-lora --pause-tokens 0
python scripts/eval_when2call.py --model devops-thiago/classone-qwen3.5-9b --limit 100 --pause-tokens 0
python scripts/eval_prompt_injection.py --model devops-thiago/classone-qwen3.5-9b --pause-tokens 0
```

---

## Reference & Deep Dives

- **Architecture Specification:** [`ARCHITECTURE.md`](ARCHITECTURE.md)
- **Empirical Learnings & Plateau Analysis:** [`docs/empirical_learnings_and_plateau_analysis.md`](docs/empirical_learnings_and_plateau_analysis.md)
- **Docker CPU Benchmarks:** [`docs/docker_cpu_benchmarks.md`](docs/docker_cpu_benchmarks.md)
- **Hugging Face Collection:** [`devops-thiago/classone-system-1-decision-models`](https://huggingface.co/collections/devops-thiago/classone-system-1-decision-models-6ac685b81e8bbfa51e646e21)
- **Training Dataset Release:** [`devops-thiago/classone-system-one-curriculum`](https://huggingface.co/datasets/devops-thiago/classone-system-one-curriculum)

---

## License & Attribution

- **Source Code & Python SDK (`src/`, `scripts/`):** Released under the **[Apache License 2.0](LICENSE)** (permissive commercial and open-source use).
- **Training Curriculum Dataset ([`devops-thiago/classone-system-one-curriculum`](https://huggingface.co/datasets/devops-thiago/classone-system-one-curriculum)):** Released under **[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)** (NonCommercial ShareAlike) because it incorporates upstream non-commercial academic and safety benchmark subsets. Review upstream dataset licenses before commercial retraining.
- **Base Models:** **Gemma** is a trademark of Google LLC and provided subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms); **Qwen 3.5** is developed by the Qwen Team at Alibaba Cloud and provided under the [Apache 2.0 License](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/LICENSE).
