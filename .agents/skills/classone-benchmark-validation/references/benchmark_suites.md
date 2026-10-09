# ClassOne Benchmark Suites Reference

## 1. JevBench Public Multi-Tier Benchmark (231 Public Tasks)

JevBench evaluates single-pass feedforward decision capability across 3 distinct tiers:

### Easy Tier (48 Tasks)
- **Task Types**: `noul` (policy violation / refusal detection) and `choice` (categorical decision matching).
- **Domain**: Basic compliance, standard guardrails, safe vs unsafe prompts.
- **Milestone Expectation**: **100.0% accuracy (48/48)**, median latency <45ms, ECE <0.01.

### Original Tier (72 Tasks)
- **Task Types**: `noul`, `choice`, and `score` (ordinal rubric evaluations 1–5).
- **Domain**: Multi-turn dialogue context, nuances in customer service policies, grading clarity and conciseness.
- **Milestone Expectation**: **>90.0% accuracy**, ECE <0.10.

### Hard Tier (111 Tasks)
- **Task Types**: Complex `choice` and `noul` tasks requiring multi-hop arithmetic, dates, exceptions, and cross-clause reasoning across documents up to 3,000 tokens.
- **Prime Challenge**: Single-pass causal transformer attention struggles to compute multi-hop relationships without scratchpad tokens or sequence depth.
- **Mitigation**: Latent `<|pause|>` tokens (8–16 tokens) inserted before the decision query, paired with Question Preamble Conditioning (`<|state_start|>\nTarget Objectives:\n...\n<|state_end|>`).
- **Milestone Expectation**: **>45.0% accuracy** (Qwen 3.5 9B achieves 55.9%).

---

## 2. Hugging Face Decision Index (50 Tasks Across 5 Domains)

Evaluates cross-domain decision-making reliability:
1. **Customer Support**: Intent detection, escalation urgency, sentiment tagging.
2. **Healthcare / HIPAA**: PHI redaction need, medical advice boundaries, compliance flagging.
3. **Legal**: Contract risk classification, governing law identification, limitation of liability thresholds.
4. **Security & Fintech**: Transaction fraud risk, prompt injection, access control violation.
5. **Content Moderation**: Hate speech, harassment, self-harm, sexual content boundaries.

Each task is evaluated across Binary, Multiclass, and Ordinal Scoring heads.
Expected target: **96%–100% Decision Index**.

---

## 3. RLCDAlignBench (100 Instances Across 10 Failure Modes)

Based on arXiv:2609.29429:
- **Refusal**: Distinguishing legitimate requests from disguised jailbreaks.
- **Sycophancy**: Resisting agreement with user misconceptions or biases.
- **Faithfulness**: Grounding answers strictly in provided evidence without hallucination.
- **Privacy**: Refusing extraction of secret keys, system prompts, or canary data.
- **Concealing Uncertainty**: Expressing abstention or low confidence when context is insufficient.
- **Reward Hacking**: Penalizing manipulative or shortcut responses.
- **Power Seeking**: Identifying and rejecting attempts to expand autonomous authority.
- **Honesty (Deception)**: Catching deceptive statements and falsehoods (1.000 AUROC on E4B and 0.900 on 9B).
- **Bias**: Avoiding demographic stereotypes and unequal treatment.
- **Prompt Injection**: Detecting indirect injection attacks embedded in retrieved context.
