# Implementation Plan v2: Targeted Accuracy Enhancements & Long-Context Routing

## Executive Summary & Diagnostic Grounding
Plan v1 demonstrated that:
1. **Generic Label Tags Corrupt ChoiceHead:** When external datasets used superficial labels (`"A": "Option A"`, `"opt_1"`), `ChoiceHead`'s bilinear interaction weights ($[q; k; |q - k|; q \odot k]$) collapsed from 97.9% down to 33.3% because $k$ contained zero semantic information. Choice criteria must **always** be rich natural language descriptions.
2. **Causal Attention Blindness on Long Contexts:** In JevBench Hard Tier (avg. 1,100 tokens, up to 3,600 tokens), placing questions strictly *after* 3,000 tokens of contracts causes the query token to lose track of fine-grained numerical thresholds and amendment clauses.
3. **4D Masking Did Not Address Context Bottlenecks:** Custom 4D option isolation does not resolve the "lost-in-the-middle" problem for multi-clause contracts.

Plan v2 directly addresses these root causes with:
- **Semantic Text Extraction:** Parsing actual natural language option texts for Sycophancy (Anthropic) and Faithfulness (HaluEval).
- **Question Preamble Conditioning:** Placing question intent *before* long context states so causal attention heads track relevant clauses as the document is ingested.
- **Preserved 6-Domain Core:** Retaining the exact data distribution that achieved 97.9% Easy tier and 56.9% Original tier.

---

## Technical Specifications

### 1. Semantic Criteria Ingestion (Fixing Sycophancy & Faithfulness)
Rather than passing raw letters (`"A"`, `"B"`), the data ingestion script extracts the actual statement text from prompts:

| Dataset | Failure Mode | Raw Format | Semantic Mapping Fix | Target Selection |
|---|---|---|---|---|
| **`Anthropic/model-written-evals`** | Sycophancy (0.200 AUROC) | Prompt contains `(A) [Statement 1]` and `(B) [Statement 2]` | Regex-parse statement text: `criteria={"truthful": stmt_truth, "sycophantic": stmt_flattery}` | Model trains to match `truthful` and reject `sycophantic` |
| **`pminervini/HaluEval`** | Faithfulness (0.400 AUROC) | `knowledge`, `question`, `right_answer`, `hallucinated_answer` | `criteria={"factual": right_answer, "unsupported": hallucinated_answer}` | Model trains to match `factual` grounded in reference knowledge |
| **`coastalcph/lex_glue`** | Hard Tier Contracts | Legal clause text + category label | `criteria={c: "Legal provision governing " + c.lower()}` with 4 sampled candidate covenants | Model learns contract clause classification |

### 2. Long-Context Prompt Architecture: Question Preamble Conditioning
For inputs where `state` exceeds 800 tokens (e.g. multi-page procurement policies and insurance forms):
- **Standard Layout:** `<|state_start|> [3,000 tokens of policy] <|state_end|> <|choice_start|> [Question + Options] <|choice_end|>`
  - *Problem:* In causal attention, tokens in the policy cannot attend to the question because the question appears in the future.
- **Preamble Conditioning Layout:**
  `<|instruction_start|> Focus on: [Question Instructions] <|instruction_end|> <|state_start|> [Policy Document] <|state_end|> <|choice_start|> [Criteria Options] <|choice_end|>`
  - *Mechanism:* Every token in the 3,000-token contract can causally attend to the question instructions, priming attention heads to extract relevant thresholds, dollar amounts, and exclusion clauses.

---

## Implementation Steps

### Phase 1: Data Pipeline Upgrade (`scripts/fetch_advanced_datasets.py`)
1. Implement regex extractor for Anthropic Sycophancy to extract full natural language options (`stmt_a`, `stmt_b`).
2. Map HaluEval samples directly to `{"factual": right_answer, "unsupported": hallucinated_answer}`.
3. Build a balanced 7,000-sample corpus:
   - 1,200 SNLI (NLI & Factuality)
   - 1,200 Banking77 (Intent Categorization, 4 options each)
   - 1,200 BeaverTails (Safety & Refusal)
   - 1,000 Agentic Injections (Security & Tool Hijacking)
   - 1,000 SciQ (Scientific Abstention, 4 options each)
   - 800 LexGLUE LEDGAR (Legal Contract Provisions)
   - 600 Anthropic Sycophancy (Semantic Statement Pairs)
4. Export to `data/training_corpus.jsonl`.

### Phase 2: Tokenizer Preamble Conditioning (`src/classone/tokenizer.py`)
1. In `ClassOnePromptBuilder.pack`:
   - If total prompt length or state length is long (>800 tokens), prepend `<|instruction_start|>\n[Question Instructions]\n<|instruction_end|>\n` before `<|state_start|>`.
   - Maintain standard prompt formatting for standard short inputs (<800 tokens) to guarantee zero regression on Easy and Original tiers.
2. Keep all special tokens aligned with published vocabulary.

### Phase 3: Dual-GPU Distributed Training & Calibration
1. Launch distributed fine-tuning across 2x RTX 5060 Ti GPUs:
   ```powershell
   python scripts/train_rlcd.py --data data/training_corpus.jsonl --gpus 2 --epochs 1 --batch-size 2 --lr 2e-4
   ```
2. Gradient checkpointing enabled (`use_reentrant=False`), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
3. Clamped temperature calibration: $T_{\text{noul}} \in [0.45, 0.60]$, $T_{\text{choice}} \in [0.45, 0.60]$, $T_{\text{score}} \in [0.55, 0.65]$.

### Phase 4: Full Multi-Tier Validation
1. Merge weights to `checkpoints/classone_gemma4_e2b_merged`.
2. Run `scripts/eval_jevbench_all_tiers.py`:
   - Verify Easy Tier remains $\ge 95\%$.
   - Verify Original Tier remains $\ge 55-60\%$.
   - Measure Hard Tier accuracy improvement (target: $\ge 45-50\%$, up from 33.3%).
3. Run `scripts/eval_rlcd_alignbench.py`:
   - Verify Sycophancy AUROC improves from 0.200 to $\ge 0.55-0.65$.
   - Verify Faithfulness accuracy improves from 40% to $\ge 60\%$.
   - Verify Concealing Uncertainty and Honesty remain $\ge 0.65$ AUROC.

---

## Verification & Safety Boundaries
- **Zero Regression Safeguard:** If Easy Tier drops below 90% at any point during validation, abort and revert to the verified checkpoint.
- **Memory Safety:** Batch size 2 with gradient checkpointing strictly limits VRAM to ~7 GB per GPU (well within 16 GB capacity).
- **Publishing Standard:** Only unquantized safetensors shards and calibrated decision heads will be pushed upon validation.

---

## Phase 5: Ambiguity Handling & Zero-Confidence Optimization

### The Problem
When the model receives an ambiguous or out-of-distribution (OOD) input (e.g., asking a billing vs. tech question on a weather-related text), the bilinear `ChoiceHead` correctly fails to find semantic similarity to any option. The raw probabilities collapse to a uniform distribution ($P \approx 1/K$), which correctly drives the scale-invariant confidence formula ($\frac{K \cdot P_{\max} - 1}{K - 1}$) to **0.0**. However, the model still arbitrarily forces a choice selection via `argmax(probs)`, leaving the burden of ambiguity handling entirely on downstream SDK consumers.

### Objective
Improve the model's robustness and API ergonomics when dealing with unanswerable or ambiguous scenarios, ensuring the engine explicitly signals OOD inputs rather than relying solely on the downstream `confidence == 0.0` check.

### Investigated Solutions

#### 1. Dynamic Implicit Abstention (The "Null" Option)
- **Mechanism:** Automatically inject a latent `"<|abstain|>": "Information is ambiguous, irrelevant, or not present"` option into the criteria dict during `ClassOnePromptBuilder.pack()` for all `ChoiceQuestion`s. 
- **Benefit:** Transforms a $K$-way forced choice into a $(K+1)$-way choice. If the input is OOD, the model explicitly selects the `<|abstain|>` option with high confidence rather than outputting 0.0 confidence across the actual candidates.
- **SDK Impact:** The SDK can intercept the `<|abstain|>` key and expose an `is_decisive: bool` or `is_ambiguous: bool` property on `ChoiceResult`.

#### 2. Soft-Label Entropy Training (Uniform Target Regularization)
- **Mechanism:** Modify `RLCDLoss.forward_multiclass` to accept soft targets instead of strictly one-hot targets. When compiling datasets (like `SQuAD 2.0 Unanswerable` or `SciQ-Uncertainty`), if an item is marked as unanswerable, set the target tensor to a uniform distribution $[1/K, 1/K, \dots, 1/K]$.
- **Benefit:** Actively teaches the `ChoiceHead` to pull all logits together (minimizing margin) when confronted with OOD text, rather than letting the weights drift arbitrarily. This guarantees the normalized confidence evaluates perfectly to `0.0`.

#### 3. SDK Ergonomics (Immediate Mitigation)
- **Mechanism:** Update the `classone.sdk.ChoiceAnswer` schema to include explicit helper properties:
  - `is_decisive: bool` (e.g., `self.confidence >= 0.15`)
  - `margin: float` (difference between top 1 and top 2 probabilities)
- **Benefit:** Developers don't have to manually interpret why `confidence = 0` happened; they can simply route logic via `if not response["intent"].is_decisive: escalate_to_human()`.
