# ClassOne: System 1 Decision Model Architecture

[![CI](https://github.com/devops-thiago/class-one/actions/workflows/ci.yml/badge.svg)](https://github.com/devops-thiago/class-one/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/devops-thiago)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Curriculum%20(23.5k)-orange)](https://huggingface.co/datasets/devops-thiago/classone-system1-decision-curriculum)

**ClassOne** is an open-source **System 1 decision model architecture**. Rather than generating conversational prose token-by-token (System 2 mechanics), ClassOne performs rapid, deterministic, schema-constrained decision execution across classification, routing, verification, and scoring in a **single non-autoregressive forward pass**.

Inspired by Daniel Kahneman's cognitive framework in *Thinking, Fast and Slow* and strictly proper scoring rules (Brier, 1950; Gneiting & Raftery, 2007).

---

## Benchmark Results

### Official Third-Party Evaluation (Benchmark Heaven)

Independent third-party evaluation is hosted on **[Benchmark Heaven](https://benchmarkheaven.com/jev-models)** ([`classone-qwen3.5-9b`](https://benchmarkheaven.com/jev-models/classone-qwen3.5-9b), benchmark version `v1.8.8`, checked `2026-11-13`):

| Model | Composite Score | Intelligence | Calibration | Capability | Source |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`classone-qwen3.5-9b`** | **43.0** (Rank **#113**) | **42.2** | **58.7** | **50.5** (**#155** of 217 open-weights) | [`benchmarkheaven.com/jev-models/classone-qwen3.5-9b`](https://benchmarkheaven.com/jev-models/classone-qwen3.5-9b) |

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
   Combines scaled dot-product attention with full non-linear bilinear matching:
   $$\text{features} = [q;\, k;\, |q - k|;\, q \odot k], \quad \text{logits} = \frac{q \cdot k}{\sqrt{d}} + \text{MLP}(\text{features})$$
   Enables deep interaction matching between prompt instructions and semantic candidate descriptions without sequence decode.
2. **Question Preamble Conditioning:**
   For documents $>600$ tokens, the prompt builder automatically injects `<|state_start|>\nTarget Objectives:\n...\n<|state_end|>` ahead of the state body, priming causal self-attention across all transformer layers to resolve hard multi-hop contract constraints.
3. **2D / 4D Position-Invariant Attention Mask:**
   Eliminates option-order recency bias by isolating candidate options during ingestion so choices attend only to the state and criteria without causally attending to earlier options.
4. **Scale-Invariant Normalized Confidence:**
   Computes calibrated confidence invariant to candidate set size:
   $$\text{Confidence} = \frac{K \cdot P_{\max} - 1}{K - 1} \quad \text{for } K \text{ choices}$$
5. **Precision Guard & Dynamic Quantization:**
   Backbone weights support 4-bit NF4 and 8-bit dynamic quantization while decision heads remain strictly in **FP16 / FP32** to preserve calibrated probability margins.
6. **Ultra-Low Latency CPU Containerization:**
   Containerized execution on **2 vCPUs and 8GB RAM** achieves **2,097.15 tokens/sec** and **19.54 decisions/sec** at 754.6 ms median latency (0 GPUs, 0 GGUF).

---

## Training Methodology: RLCD (Reinforcement Learning for Calibrated Decisions)

ClassOne models are aligned using **Strictly Proper Scoring Rules** rather than open-ended conversational RLHF:

1. **Multi-Objective RLCD Loss:**
   $$\mathcal{L}_{\text{RLCD}} = \alpha \mathcal{L}_{\text{Focal}}(\gamma=2.0) + \beta \mathcal{L}_{\text{Brier}} + \lambda \mathcal{L}_{\text{Margin}}$$
   - **Focal Log-Loss:** Down-weights easy examples and forces gradient updates to concentrate on hard decision boundaries.
   - **Brier Score:** Penalizes probability deviation, mathematically guaranteeing that the model has zero incentive to hedge or hallucinate unwarranted overconfidence.
   - **Contrastive Margin Penalty:** Directly penalizes decisions with sub-threshold logit separation between top-1 and distractor options.
2. **Temperature Floor Clamping:**
   Post-hoc temperature scaling prevents probability softening by enforcing empirical calibration floors:
   $$T_{\text{noul}} \in [0.45, 0.60], \quad T_{\text{choice}} \in [0.45, 0.60], \quad T_{\text{score}} \in [0.55, 0.65]$$
3. **Progressive Multi-Stage Curricula:**
   Trained on a 23,500-sample human-curated curriculum spanning ContractNLI, CUAD, SciQ, Banking77, ToxicChat, XSTest, and Agentic boundary pairs.
4. **Weight-Averaged Model Souping:**
   Combines converged LoRA weights and decision heads across curriculum stages (`soup_heads_70.py`), yielding monotonic calibration and benchmark gains.

---

## Published Models on Hugging Face Hub

> [!TIP]
> Explore all published models and datasets in the official Hugging Face Collection:
> **[ClassOne System 1 Decision Models Collection](https://huggingface.co/collections/devops-thiago/classone-system-1-decision-models-6ac685b81e8bbfa51e646e21)**

| Repository ID | Base Model | Context | Calibrated Heads SHA256 (Verified on Hub) | Formats |
| :--- | :--- | :---: | :--- | :--- |
| [`devops-thiago/classone-qwen3.5-9b`](https://huggingface.co/devops-thiago/classone-qwen3.5-9b) | Qwen/Qwen3.5-9B | 256k | `68FCDB3106BD5026B7CAFF507293961C16BD6BA178F74E9CFBF6266317C2A141` | BF16 Standalone, 4-bit NF4 |
| [`devops-thiago/classone-qwen3.5-4b`](https://huggingface.co/devops-thiago/classone-qwen3.5-4b) | Qwen/Qwen3.5-4B | 256k | `654DA038EA5883916F3323F1A5BE64DEEFADB864B2EAE48DF9CD915D9C8A8F66` | BF16 Standalone, 4-bit NF4 |
| [`devops-thiago/classone-gemma4-e4b`](https://huggingface.co/devops-thiago/classone-gemma4-e4b) | google/gemma-4-E4B-it | 128k | `8F7157F4D67D21698F94A3DF69B8492F4CC592DAC65E91AC47A83FD2E083AFDC` | BF16 Standalone, 4-bit NF4 |
| [`devops-thiago/classone-qwen3.5-2b`](https://huggingface.co/devops-thiago/classone-qwen3.5-2b) | Qwen/Qwen3.5-2B | 256k | `BD24D8639DB9054CD9E0869E8DD2CF432D0B9A63AC82C557186A95D371E255F6` | FP16/BF16 Standalone, 8-bit |
| [`devops-thiago/classone-gemma4-e2b`](https://huggingface.co/devops-thiago/classone-gemma4-e2b) | google/gemma-4-e2b-it | 128k | `F6A6E42365158D99807B8995C800429A0BD2B12755EDC1DEC5823400D35F337D` | FP16/BF16 Standalone, 8-bit |

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/devops-thiago/class-one.git
cd class-one
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Run Test Suite

```bash
pytest tests/ -v
```

### 3. Start API Service

```bash
uvicorn classone.server.app:app --host 0.0.0.0 --port 8000
```

By default, the server derives the per-request prompt token limit directly from the loaded model's configuration (`text_config.max_position_embeddings` or `max_position_embeddings`, e.g., `131072` for Gemma 4 E2B/E4B and `262144` for Qwen 3.5). To enforce a tighter cap for 16 GB VRAM or latency budgets, set `CLASSONE_MAX_TOKENS_PER_REQUEST` (e.g., `CLASSONE_MAX_TOKENS_PER_REQUEST=8192`).

### 4. Query via Python SDK

```python
from classone import ClassOneClient, Choice, Noul, Score

with ClassOneClient(base_url="http://localhost:8000") as client:
    response = client.decide(
        state="Customer account was locked after three incorrect attempts within 15 minutes.",
        questions={
            "urgent": Noul(instructions="Is this an urgent security event?"),
            "action": Choice(
                instructions="Action required:",
                criteria={
                    "unlock": "Send automated unlock verification link",
                    "escalate": "Escalate to SecOps fraud team",
                },
            ),
            "severity": Score(instructions="Assess severity:", criteria=["low", "medium", "critical"]),
        },
    )

    print("Urgent P(true):", response.nouls["urgent"].noul)
    print("Action Choice:", response.choices["action"].choice)
    print("Choice Margin:", response.choices["action"].margin)
    print("Severity Score:", response.scores["severity"].score)
```

---

## Docker Deployment (2 vCPUs, 8GB RAM, Pure CPU)

Build and deploy the lean CPU container with zero GPU requirements:

```bash
# 1. Build image (uses CPU-only PyTorch wheels)
docker build -t classone:cpu .

# 2. Run container constrained to 2 vCPUs and 8GB RAM
docker run -d --name classone-cpu \
  --cpus=2 --memory=8g -p 8000:8000 \
  -e CLASSONE_DEVICE=cpu \
  -e CLASSONE_BASE_MODEL=/app/checkpoints/classone_cpu_q8.pt \
  -v "$(pwd)/checkpoints:/app/checkpoints:ro" \
  classone:cpu

# 3. Run parallel load benchmark (5 concurrent worker threads)
python scripts/benchmark_docker_cpu.py --url http://localhost:8000 --concurrency 5 --requests 50
```

---

## Evaluation & Benchmark Reproduction

### 1. Self-Run JevBench v1.6.1 Reproduction (Public Tasks)

Evaluates chance-corrected Intelligence (equal type weighting across Choice, Noul, Score), multi-bin Expected Calibration Error, Speed, and local cost estimation across the 231 public JevBench tasks:

```bash
# Evaluate Qwen 3.5 9B locally on GPU:
python scripts/eval_jevbench_v16.py --mode local

# Evaluate against live HTTP API endpoint:
python scripts/eval_jevbench_v16.py --mode api --endpoint http://127.0.0.1:8000/v1/decide
```

**Self-Measured Public 4-Axis Scorecard (`2x NVIDIA RTX 5060 Ti 16 GB` · Date: `2026-11-13` · Commit: `c375207`, ClassOne Qwen 3.5 9B):**

| Axis | Weight | Score (0–100) | Metric Details |
| :--- | :---: | :---: | :--- |
| **Composite Score (Self-Run)** | **100%** | **73.48** | Equal-weight 4-axis Harmonic Mean (see Benchmark Heaven above for official `v1.8.8` scores) |
| **Capability Score** | — | **75.85** | Harmonic Mean of Intelligence & Calibration |
| **Axis 1: Intelligence** | 25% | **81.38** | Choice: 85.07 \| Noul: 84.07 \| Score: 75.00 |
| **Axis 2: Calibration** | 25% | **71.02** | ECE: 0.1549 \| Brier: 0.1596 |
| **Axis 3: Speed** | 25% | **89.01** | p50: 148.5 ms \| p95: 845.0 ms |
| **Axis 4: Cost** | 25% | **59.41** | $0.02253 per 1,000 decisions |

*Public Tiers: Easy 100.0% (48/48), Standard 97.2% (70/72), Hard 56.8% (63/111).*

---

### 2. Decision Index Evaluation Suite (`multimodalart/jev-decision-index`)

Self-run evaluation across all 5 capability domains using the public Decision Index task set (`scores.json` and `index.json` generated locally):

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

---

### 3. Additional Benchmark Suites & Reproduction Commands

All benchmark evaluation scripts support both local checkpoint paths and Hugging Face Hub identifiers:

```bash
# 1. JevBench v1.6.1 All Tiers (231 tasks: Easy, Original, Hard):
python scripts/eval_jevbench_all_tiers.py devops-thiago/classone-qwen3.5-9b --no-lora --pause-tokens 0

# 2. Decision Index Suite (5 domains, local GPU):
python scripts/eval_decision_index.py --mode local --model-path devops-thiago/classone-qwen3.5-9b --quant none

# 3. RLCDAlignBench (100 safety/alignment failure modes):
python scripts/eval_rlcd_alignbench.py devops-thiago/classone-qwen3.5-9b --no-lora --pause-tokens 0

# 4. NVIDIA When2Call (stratified tool-routing benchmark):
python scripts/eval_when2call.py --model devops-thiago/classone-qwen3.5-9b --limit 100 --pause-tokens 0

# 5. Deepset Prompt Injections (adversarial injection defense):
python scripts/eval_prompt_injection.py --model devops-thiago/classone-qwen3.5-9b --pause-tokens 0

# Verify all published Hugging Face models via SDK:
python scripts/test_all_hf_models_sdk.py
```

---

## Reference & Deep Dives

- **Architecture Specification:** [`ARCHITECTURE.md`](ARCHITECTURE.md)
- **Empirical Learnings & Plateau Analysis:** [`docs/empirical_learnings_and_plateau_analysis.md`](docs/empirical_learnings_and_plateau_analysis.md)
- **Docker CPU Benchmarks:** [`docs/docker_cpu_benchmarks.md`](docs/docker_cpu_benchmarks.md)
- **Hugging Face Collection:** [`devops-thiago/classone-system-1-decision-models`](https://huggingface.co/collections/devops-thiago/classone-system-1-decision-models-6ac685b81e8bbfa51e646e21)
- **Training Dataset Release:** [`devops-thiago/classone-system-one-curriculum`](https://huggingface.co/datasets/devops-thiago/classone-system-one-curriculum)

---

## Client SDKs & Package Distribution

ClassOne provides official client libraries across 6 languages with native HTTP transports, zero unnecessary dependencies, and type-safe probabilistic primitives (`Noul`, `Choice`, `Score`):

| Language | Package Registry | Installation / Dependency | Repository |
|---|---|---|---|
| **Python** | [PyPI (`classone`)](https://pypi.org/project/classone/) | `pip install classone` | Built-in (`src/classone/sdk`) |
| **Node.js / TS** | [npm (`@classone/sdk`)](https://www.npmjs.com/package/@classone/sdk) | `npm install @classone/sdk` | [`classone-sdks/nodejs`](https://github.com/devops-thiago/classone-sdks/tree/main/nodejs) |
| **Go** | [pkg.go.dev](https://pkg.go.dev/github.com/devops-thiago/classone-sdks/go) | `go get github.com/devops-thiago/classone-sdks/go@v0.1.1` | [`classone-sdks/go`](https://github.com/devops-thiago/classone-sdks/tree/main/go) |
| **Rust** | [crates.io (`classone`)](https://crates.io/crates/classone) | `cargo add classone` | [`classone-sdks/rust`](https://github.com/devops-thiago/classone-sdks/tree/main/rust) |
| **Java** | [Maven Central (`io.classone:classone-sdk`)](https://central.sonatype.com/artifact/io.classone/classone-sdk) | `implementation 'io.classone:classone-sdk:0.1.1'` | [`classone-sdks/java`](https://github.com/devops-thiago/classone-sdks/tree/main/java) |
| **Ruby** | [RubyGems (`classone`)](https://rubygems.org/gems/classone) | `gem install classone` | [`classone-sdks/ruby`](https://github.com/devops-thiago/classone-sdks/tree/main/ruby) |

To verify all SDKs against a running server:
```bash
python scripts/test_all_sdks_live.py --url http://127.0.0.1:8000
python scripts/package_and_verify_all_sdks.py
```

---

## License & Attribution

- Released under the [Apache 2.0 License](LICENSE).
- **Gemma** is a trademark of Google LLC and provided subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
- **Qwen** is developed by the Qwen Team at Alibaba Cloud and provided under the [Apache 2.0 License](https://github.com/QwenLM/Qwen2.5).
