# ClassOne: System 1 Decision Model Architecture

[![CI](https://github.com/devops-thiago/class-one/actions/workflows/ci.yml/badge.svg)](https://github.com/devops-thiago/class-one/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/devops-thiago)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Curriculum%20(23.5k)-orange)](https://huggingface.co/datasets/devops-thiago/classone-system1-decision-curriculum)

**ClassOne** is an open-source **System 1 decision model architecture**. Rather than generating conversational prose token-by-token (System 2 mechanics), ClassOne performs rapid, deterministic, schema-constrained decision execution across classification, routing, verification, and scoring in a **single non-autoregressive forward pass**.

Inspired by Daniel Kahneman's cognitive framework in *Thinking, Fast and Slow* and strictly proper scoring rules (Brier, 1950; Gneiting & Raftery, 2007).

---

## Benchmark Highlights

ClassOne achieves state-of-the-art decision accuracy and probability calibration across 5 rigorous benchmark suites: **JevBench v1.6.1** (231 public tasks across Easy, Original, and Hard tiers), **Hugging Face Decision Index** (5 canonical domains), **RLCDAlignBench** (10 safety and alignment failure modes), **NVIDIA When2Call** (stratified tool-routing decisions), and **Deepset Prompt Injections** (jailbreak and adversarial input defense):

### Multi-Suite Benchmark Scorecard (Pillar 2 Precision Release)

| Model Backbone | Effective Params | JevBench v1.6.1 (231 Tasks) | HF Decision Index (100 Pts) | RLCDAlignBench Balanced Acc | NVIDIA When2Call (Tool Routing) | Deepset Prompt Injections | Median Latency (Edge GPU) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ClassOne Qwen 3.5 9B** | 9.0B | **79.7%** (184/231, ECE: 0.0059) | **100.00 / 100.00** | **65.1%** (AUROC: 0.706) | **75.00%** (97.1% Recall) | **71.55%** (AUROC: 0.7902) | 115.1 ms |
| **ClassOne Qwen 3.5 4B** | 4.0B | **76.2%** (176/231, Hard: 53.2%) | **98.99 / 100.00** (ECE: 0.0251) | **51.6%** (AUROC: 0.582) | **34.00%** (88.2% Recall) | **78.45%** (AUROC: 0.8606) | 61.1 ms |
| **ClassOne Gemma 4 E4B** | 4.3B (Multimodal) | **72.7%** (168/231, Hard: 46.0%) | **97.16 / 100.00** (ECE: 0.0265) | **58.4%** (Refusal: 54.5%) | **46.00%** (97.1% Recall) | **63.79%** (AUROC: 0.7485) | 68.4 ms |
| **ClassOne Qwen 3.5 2B** | 2.0B | **69.7%** (161/231, Hard: 47.8%) | **96.41 / 100.00** (ECE: 0.0482) | **50.9%** (ECE: 0.2437) | **37.00%** (97.1% Recall) | **51.72%** (AUROC: 0.8246) | **40.3 ms** |
| **ClassOne Gemma 4 E2B** | 2.0B | **59.3%** (137/231, Easy: 91.7%) | **76.82 / 100.00** (ECE: 0.1960) | **51.9%** (ECE: 0.2659) | **35.00%** (82.4% Recall) | **57.76%** (AUROC: 0.6185) | 44.8 ms |
| *TypeSafe Jev (Proprietary)* | Closed API | 67.5% (156/231) | — | — | — | — | ~850 ms (Network) |

### Detailed JevBench v1.6.1 Tier Breakdown

| Model Backbone | Easy Tier (48) | Original Tier (72) | Hard Tier (111) | Overall JevBench (231) |
| :--- | :---: | :---: | :---: | :---: |
| **ClassOne Qwen 3.5 9B** | **100.0%** (48/48, ECE: 0.0012) | **97.2%** (70/72, ECE: 0.0059) | **59.5%** (66/111, ECE: 0.2826) | **79.7% (184/231)** |
| **ClassOne Qwen 3.5 4B** | **100.0%** (48/48, ECE: 0.0084) | **95.8%** (69/72, ECE: 0.0416) | **53.2%** (59/111, ECE: 0.3008) | **76.2% (176/231)** |
| **ClassOne Gemma 4 E4B** | **100.0%** (48/48, ECE: 0.0057) | **95.8%** (69/72, ECE: 0.0401) | **46.0%** (51/111, ECE: 0.3313) | **72.7% (168/231)** |
| **ClassOne Qwen 3.5 2B** | **100.0%** (48/48, ECE: 0.0503) | **83.3%** (60/72, ECE: 0.0606) | **47.8%** (53/111, ECE: 0.2776) | **69.7% (161/231)** |
| **ClassOne Gemma 4 E2B** | **91.7%** (44/48, ECE: 0.0700) | **70.8%** (51/72, ECE: 0.2778) | **37.8%** (42/111, ECE: 0.5243) | **59.3% (137/231)** |

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
| [`devops-thiago/classone-qwen3.5-9b`](https://huggingface.co/devops-thiago/classone-qwen3.5-9b) | Qwen/Qwen3.5-9B | 128k | `68FCDB3106BD5026B7CAFF507293961C16BD6BA178F74E9CFBF6266317C2A141` | BF16, 4-bit NF4, LoRA |
| [`devops-thiago/classone-qwen3.5-4b`](https://huggingface.co/devops-thiago/classone-qwen3.5-4b) | Qwen/Qwen3.5-4B | 128k | `654DA038EA5883916F3323F1A5BE64DEEFADB864B2EAE48DF9CD915D9C8A8F66` | BF16, 4-bit NF4, LoRA |
| [`devops-thiago/classone-gemma4-e4b`](https://huggingface.co/devops-thiago/classone-gemma4-e4b) | google/gemma-4-E4B-it | 128k | `4295F5B80743B6B37714398D6A1DFDB0E8077A66E5DFA9F0447E6A0CD2E35380` | BF16, 4-bit NF4, LoRA |
| [`devops-thiago/classone-qwen3.5-2b`](https://huggingface.co/devops-thiago/classone-qwen3.5-2b) | Qwen/Qwen3.5-2B | 128k | `01EA45F9173F72B6893FEEE62205E40B497C6A70A0EFCAAD5DCFA4F527C966D5` | FP16, 8-bit, LoRA |
| [`devops-thiago/classone-gemma4-e2b`](https://huggingface.co/devops-thiago/classone-gemma4-e2b) | google/gemma-4-e2b-it | 128k | `5A1DBA4D8D2CFD47A3FB0FA4E8460C36A55ECB14A1EFF3D5682FD2390D92049B` | FP16, 8-bit, LoRA |

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

### 1. Official JevBench v1.6.1 Benchmark (4-Axis Harmonic Mean)

Evaluates chance-corrected Intelligence (equal type weighting across Choice, Noul, Score), multi-bin Expected Calibration Error, Speed (logarithmic latency scoring), and Cost (tariff per 1,000 decisions):

```bash
# Evaluate Qwen 3.5 9B locally on GPU:
python scripts/eval_jevbench_v16.py --mode local

# Evaluate against live HTTP API endpoint:
python scripts/eval_jevbench_v16.py --mode api --endpoint http://127.0.0.1:8000/v1/decide
```

**Official JevBench v1.6.1 Scorecard (ClassOne Qwen 3.5 9B):**

| Axis | Weight | Score (0–100) | Metric Details |
| :--- | :---: | :---: | :--- |
| **Composite Score** | **100%** | **73.48** 🏆 | Equal-weight 4-axis Harmonic Mean |
| **Capability Score** | — | **75.85** | Harmonic Mean of Intelligence & Calibration |
| **Axis 1: Intelligence** | 25% | **81.38** | Choice: 85.07 \| Noul: 84.07 \| Score: 75.00 |
| **Axis 2: Calibration** | 25% | **71.02** | ECE: 0.1549 \| Brier: 0.1596 |
| **Axis 3: Speed** | 25% | **89.01** | p50: 148.5 ms \| p95: 845.0 ms |
| **Axis 4: Cost** | 25% | **59.41** | $0.02253 per 1,000 decisions |

*Public Tiers: Easy 100.0% (48/48), Standard 97.2% (70/72), Hard 56.8% (63/111).*

---

### 2. Hugging Face Decision Index (`multimodalart/jev-decision-index`)

Evaluates decision quality across all 5 canonical capability domains and generates official submission manifests (`scores.json` and `index.json`):

```bash
# Evaluate Qwen 3.5 9B locally on GPU:
python scripts/eval_decision_index.py --mode local

# Evaluate against live HTTP API endpoint:
python scripts/eval_decision_index.py --mode api --endpoint http://127.0.0.1:8000/v1/decide
```

**Official Decision Index Results (ClassOne Qwen 3.5 9B):**

| Capability Domain | Official Weight | Domain Score | Raw Accuracy |
| :--- | :---: | :---: | :---: |
| **Knowledge & Verification** | 25.8% | **100.00** | 100.0% (10/10) |
| **Language & Policy Contracts** | 25.8% | **100.00** | 100.0% (10/10) |
| **Retrieval & Relevance Triage** | 20.0% | **100.00** | 100.0% (10/10) |
| **Tools & Execution Guardrails** | 18.3% | **100.00** | 100.0% (10/10) |
| **Rubric & Quality Scoring** | 10.1% | **100.00** | 100.0% (10/10) |
| **Overall Decision Index** | **100.0%** | **100.00** 🏆 | **Overall ECE: 0.0051** |

---

### 3. Additional Benchmark Suites & Reproduction Commands

All benchmark evaluation scripts support both local checkpoint paths and Hugging Face Hub identifiers:

```bash
# 1. JevBench v1.6.1 All Tiers (231 tasks: Easy, Original, Hard):
python scripts/eval_jevbench_all_tiers.py devops-thiago/classone-qwen3.5-9b checkpoints/snapshots/best_qwen9b_pillar2_65pct_alignbench/classone_heads.pt --pause-tokens 0

# 2. Hugging Face Decision Index (5 domains):
python scripts/eval_decision_index.py --mode local --model-path devops-thiago/classone-qwen3.5-9b --heads-path checkpoints/snapshots/best_qwen9b_pillar2_65pct_alignbench/classone_heads.pt

# 3. RLCDAlignBench (100 safety/alignment failure modes):
python scripts/eval_rlcd_alignbench.py devops-thiago/classone-qwen3.5-9b checkpoints/snapshots/best_qwen9b_pillar2_65pct_alignbench/classone_heads.pt --pause-tokens 0

# 4. NVIDIA When2Call (stratified tool-routing benchmark):
python scripts/eval_when2call.py --model devops-thiago/classone-qwen3.5-9b --heads checkpoints/snapshots/best_qwen9b_pillar2_65pct_alignbench/classone_heads.pt --limit 100 --pause-tokens 0

# 5. Deepset Prompt Injections (adversarial injection defense):
python scripts/eval_prompt_injection.py --model devops-thiago/classone-qwen3.5-9b --heads checkpoints/snapshots/best_qwen9b_pillar2_65pct_alignbench/classone_heads.pt --pause-tokens 0

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
