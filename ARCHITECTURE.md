# ClassOne: System 1 Decision Model Architecture Specification

## 1. Executive Summary & Theoretical Foundations

**ClassOne** is an open-source **System 1 decision model architecture**. Rather than forcing open-ended, autoregressive token generation (System 2 mechanics) onto structured tasks, ClassOne performs rapid, deterministic, schema-constrained decision evaluation across classification, routing, verification, and scoring in a **single non-autoregressive forward pass**.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Input Sequence Representation                   │
│  <|state_start|> [Preamble] + Document State <|state_end|>             │
│  <|noul_start|> ... <|noul_end|>                                       │
│  <|choice_start|> ... <|opt_start|> K1 <|opt_end|> ... <|choice_end|> │
│  <|score_start|> ... <|level_start|> L1 <|level_end|> ... <|score_end|>│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Single Transformer Forward Pass
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │             Transformer Backbone (128k Context)         │
       │    Gemma 4 (E2B, E4B, 12B)  /  Qwen 3.5 (2B, 4B, 9B)     │
       │   [2D / 4D Block-Diagonal Position-Invariant Attention] │
       └──────────────┬──────────────────┬─────────────────┬─────┘
                      │                  │                 │
                      ▼                  ▼                 ▼
             ┌─────────────────┐ ┌───────────────┐ ┌───────────────┐
             │    NoulHead     │ │   ChoiceHead  │ │   ScoreHead   │
             │ (FP16/FP32 MLP) │ │ (Bilinear+SDP)│ │ (Ordinal MLP) │
             └────────┬────────┘ └───────┬───────┘ └───────┬───────┘
                      │                  │                 │
                      ▼                  ▼                 ▼
                 P(true) ∈ [0,1]     Option Winner    Expected Score
               Calibrated Conf     Margin & Conf     Discrete Rubric
```

### 1.1 Cognitive & Economic Foundations
1. **System 1 Thinking (Kahneman, 2011):**
   Daniel Kahneman characterizes **System 1** as fast, parallel, associative, intuitive, and probabilistic, whereas **System 2** is slow, sequential, deliberate, and rule-governed. Traditional generative LLMs force sequential token generation onto simple verification, extraction, and routing problems. ClassOne implements native System 1 computation: parallel associative pooling over representations in a single step.
2. **The Jevons Paradox (Jevons, 1865):**
   Increasing the efficiency of resource use leads to an exponential increase in total consumption. By driving decision latency down to <50 ms and token generation cost to $0, ClassOne enables autonomous multi-agent loops and enterprise services to execute thousands of decisions per transaction.

---

## 2. Core Architectural Comparison

| Dimension | Generative Autoregressive LLMs | ClassOne System 1 Architecture |
| :--- | :--- | :--- |
| **Cognitive Mode** | System 2 (sequential token rollout) | **System 1 (parallel associative decision)** |
| **Output Type** | Autoregressive string tokens | **Typed JSON schema (`Noul`, `Choice`, `Score`)** |
| **Decoding Steps** | $N$ tokens ($N$ sequential forward passes) | **1 single feedforward pass** |
| **Latency** | 1,000 – 10,000+ ms | **40 – 115 ms (GPU), ~750 ms (2 vCPU container)** |
| **Alignment Loss** | RLHF / DPO (conversational preference) | **RLCD (Proper Scoring Rules: Brier + Focal + Margin)** |
| **Structural Hallucination** | High (syntax drift, malformed JSON) | **Zero structural/syntax hallucinations** |
| **Confidence Output** | Subjective verbalized certainty | **Mathematically Calibrated Probabilities (ECE < 0.05)** |
| **Throughput (2 vCPUs)** | 10 – 30 tokens/sec | **2,097.15 tokens/sec (19.54 decisions/sec)** |

---

## 3. Mathematical Primitives & Decision Heads

Decision pointer heads (`noul_head`, `choice_head`, `score_head`) operate on the last hidden state of the transformer backbone and remain strictly in **FP16 or FP32 precision** even when backbones are quantized to 4-bit or 8-bit.

### 3.1 Noul (Boolean Verification Probability)
* **Objective:** Given state $S$ and verification instruction $I$, estimate $P(\text{true} \mid S, I) \in [0.0, 1.0]$.
* **Token Pooling:** Extract hidden representation $h_{\text{noul}} \in \mathbb{R}^d$ at the `<|noul_end|>` token delimiter.
* **Projection & Calibration:**
  $$z = W_2 \cdot \text{GELU}(W_1 h_{\text{noul}} + b_1) + b_2$$
  $$P(\text{true}) = \sigma(z / T_{\text{noul}})$$
  where $T_{\text{noul}} = \text{softplus}(w_T) + 0.1$ is clamped during post-hoc calibration to $T_{\text{noul}} \in [0.45, 0.60]$.
* **Confidence Metric:** Scale-invariant normalized confidence:
  $$\text{Confidence}_{\text{noul}} = 2 \cdot |P(\text{true}) - 0.5| \in [0.0, 1.0]$$

### 3.2 Choice (Bilinear Interaction Matching)
* **Theoretical Model:** Grounded in the Plackett-Luce choice axiom (Luce, 1959; Plackett, 1975) enhanced with deep interaction scoring.
* **Token Pooling:**
  * Query utility vector: $h_{\text{query}} \in \mathbb{R}^d$ pooled at `<|choice_end|>`.
  * Candidate option vectors: $h_{\text{opt}_k} \in \mathbb{R}^d$ pooled at `<|opt_end|>` for each choice $k \in \{1, \dots, K\}$.
* **Projection & Bilinear Interaction Scorer:**
  $$q = \text{LayerNorm}(W_q h_{\text{query}}), \quad k_i = \text{LayerNorm}(W_k h_{\text{opt}_i})$$
  Features are constructed via element-wise difference and interaction product:
  $$\text{features}_i = [q;\, k_i;\, |q - k_i|;\, q \odot k_i] \in \mathbb{R}^{4 \cdot d_{\text{proj}}}$$
  Total logit score blends scaled dot-product with MLP non-linear scoring:
  $$v_i = \frac{\langle q, k_i \rangle}{\sqrt{d_{\text{proj}}}} + \text{MLP}_{\text{scorer}}(\text{features}_i)$$
  $$p_i = \frac{\exp(v_i / T_{\text{choice}})}{\sum_{j=1}^K \exp(v_j / T_{\text{choice}})}$$
  where effective temperature is clamped to $T_{\text{choice}} \in [0.45, 0.60]$.
* **Decision Winner:** $\text{choice}^* = \arg\max_i p_i$.
* **Scale-Invariant Normalized Confidence Formula:**
  $$\text{Confidence}_{\text{choice}} = \frac{K \cdot p_{(1)} - 1}{K - 1} \in [0.0, 1.0]$$
* **Margin Metric:** Decisive probability gap between top-1 and top-2 candidate options:
  $$\Delta = p_{(1)} - p_{(2)}$$
  A decision is marked `is_decisive = True` if $\text{Confidence} \ge 0.15$ and $\Delta \ge 0.10$.

### 3.3 Score (Continuous Ordinal Rubric)
* **Objective:** Given ordered rubric levels $L_1, \dots, L_M$ ($M \in [2, 10]$), predict a continuous expected score.
* **Calculation:**
  Evaluated through `ChoiceHead` representations over level vectors $h_{\text{level}_m}$ at `<|level_end|>`:
  $$\mathbb{E}[\text{Score}] = \sum_{m=1}^M m \cdot p_m$$
  where effective temperature is clamped to $T_{\text{score}} \in [0.55, 0.65]$.
* **Confidence Metric:** Maximum probability $\max_m p_m$.

---

## 4. Sequence Packing & Attention Protocols

### 4.1 Sequence Packing Protocol
State representations and typed questions are assembled into a single sequence:

```text
<|state_start|>
Target Objectives:
- [contract_clause]: Validate indemnity cap adherence
- [severity]: Rate audit violation level
<|state_end|>
<|state_start|>
Enterprise Contract Section 14.2: Maximum indemnity is capped at $5,000,000...
<|state_end|>
<|noul_start|>
Question ID: is_compliant
Instruction: Does the contract adhere to the policy cap?
<|noul_end|>
<|choice_start|>
Question ID: clause_type
Instruction: Identify the clause structure:
<|opt_start|>Key: mutual | Criteria: Mutual indemnification<|opt_end|>
<|opt_start|>Key: unilateral | Criteria: Unilateral vendor indemnification<|opt_end|>
<|choice_end|>
```

### 4.2 Question Preamble Conditioning
In documents exceeding 600 tokens, standard causal self-attention suffers from state-ingestion distraction. `ClassOnePromptBuilder` automatically prepends:
```text
<|state_start|>
Target Objectives:
- [Q1]: instructions...
- [Q2]: instructions...
<|state_end|>
```
This primes all self-attention layers to track pertinent numbers, entities, and exception clauses during document ingestion, lifting Hard Tier accuracy above 40%.

### 4.3 2D / 4D Position-Invariant Attention Masking
To prevent candidate options from causally attending to earlier options (eliminating option-order and recency bias), ClassOne constructs a block-diagonal attention mask:
* State and preamble attend causally.
* Candidate options attend to state and instructions, but cross-option attention between candidate $i$ and candidate $j$ is masked ($-\infty$).
* Position IDs are symmetrized across all candidate option spans.

---

## 5. RLCD: Training with Strictly Proper Scoring Rules

ClassOne employs **Reinforcement Learning for Calibrated Decisions (RLCD)**, combining proper scoring rules with margin and focal regularization:

$$\mathcal{L}_{\text{RLCD}} = \alpha \mathcal{L}_{\text{Focal-NLL}} + \beta \mathcal{L}_{\text{Brier}} + \lambda \mathcal{L}_{\text{Margin}}$$

### 5.1 Loss Components
1. **Focal Log-Loss (Lin et al., 2017; Almeida, 2026):**
   $$\mathcal{L}_{\text{Focal}} = -\sum_{k=1}^K y_k (1 - p_k)^\gamma \log(p_k), \quad \gamma = 2.0$$
   Down-weights easily classified examples, forcing gradients to concentrate on difficult boundary cases.
2. **Normalized Multiclass Brier Score (Brier, 1950; Gneiting & Raftery, 2007):**
   $$\mathcal{L}_{\text{Brier}} = \frac{1}{2} \sum_{k=1}^K (p_k - y_k)^2 \in [0, 1]$$
   Strictly proper scoring rule that uniquely penalizes probability deviations from the true data generating distribution.
3. **Contrastive Margin Penalty:**
   $$\mathcal{L}_{\text{Margin}} = \max\left(0,\, m - (z_{\text{target}} - \max_{j \ne \text{target}} z_j)\right), \quad m = 1.0$$
   Explicitly widens the raw logit gap between the correct choice and the nearest distractor.

---

## 6. Multi-Scale Backbone Support

ClassOne supports a diverse family of backbones across Google Gemma 4 and Alibaba Cloud Qwen 3.5:

```
                  ┌─────────────────────────────────────────┐
                  │        ClassOne Model Portfolio         │
                  └────────────────────┬────────────────────┘
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
┌────────────────────────┐                             ┌────────────────────────┐
│     Gemma 4 Family     │                             │    Qwen 3.5 Family     │
│  google/gemma-4-E2B-it │                             │     Qwen/Qwen3.5-2B    │
│  google/gemma-4-E4B-it │                             │     Qwen/Qwen3.5-4B    │
│  google/gemma-4-12B-it │                             │     Qwen/Qwen3.5-9B    │
│ (In-Vocab PLE Mapping) │                             │ (In-Vocab <|extra_0|>) │
└────────────────────────┘                             └────────────────────────┘
```

### 6.1 In-Vocab Delimiter Mapping
- **Gemma 4:** Delimiter tokens (`<|state_start|>`, `<|state_end|>`, `<|opt_end|>`) are mapped or appended via `resize_token_embeddings`.
- **Qwen 3.5:** Custom delimiters map to pre-existing unused token IDs (`<|extra_0|>` through `<|extra_8|>`), eliminating embedding resize overhead and preserving vocabulary alignment.

---

## 7. Distributed Dual-GPU Training on Windows

To bypass WinSock TCP socket buffer overflows (`0xC0000409` in Gloo) on Windows:
- Multi-processing relies on `torch.multiprocessing.Process` with shared memory queues (`sync_gradients_shm`).
- Parameter gradients are flattened into a single contiguous buffer with zero-filling for unactivated heads, then reduced across GPU processes with sub-millisecond overhead.
- VRAM hygiene is enforced via `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and activation subgraph eviction (`del seq_hidden, hidden_states`).

---

## 8. Academic & Theoretical References

1. **Kahneman, Daniel.** (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.
2. **Brier, Glenn W.** (1950). "Verification of forecasts expressed in terms of probability". *Monthly Weather Review*, 78(1), 1–3.
3. **Gneiting, Tilmann, & Raftery, Adrian E.** (2007). "Strictly proper scoring rules, prediction, and estimation". *Journal of the American Statistical Association*, 102(477), 359–378.
4. **Guo, Chuan, Pleiss, Geoff, Sun, Yu, & Weinberger, Kilian Q.** (2017). "On calibration of modern neural networks". *ICML 2017*, PMLR 70:1321–1330.
5. **Lin, Tsung-Yi, Goyal, Priya, Girshick, Ross, He, Kaiming, & Dollár, Piotr.** (2017). "Focal loss for dense object detection". *IEEE ICCV 2017*, 2980–2988.
6. **Luce, R. Duncan.** (1959). *Individual Choice Behavior: A Theoretical Analysis*. John Wiley & Sons.
7. **Plackett, Robin L.** (1975). "The analysis of permutations". *Journal of the Royal Statistical Society: Series C (Applied Statistics)*, 24(2), 193–202.
8. **Jevons, William Stanley.** (1865). *The Coal Question*. Macmillan and Co.
