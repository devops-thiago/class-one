#!/usr/bin/env python3
"""Evaluates canonical TypeSafe AI (docs.typesafe.ai) System One patterns on ClassOne.

Tests 5 core patterns from the TypeSafe AI documentation:
1. Multi-Facet Ticket Triage (Customer Support Operations)
2. Agent Guardrails & Tool Verification ("Auto Mode")
3. Composite Scoring (Investment / Memo Evaluation)
4. Intent Routing with Confidence-Gated Branching
5. RAG Passage Quality & Citation Verification

Uses the published ClassOne model (devops-thiago/classone-gemma4-e2b) on CUDA.
"""

import time

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from classone.modeling.modeling_classone import ClassOneModel
from classone.schemas import ChoiceQuestion, NoulQuestion, ScoreQuestion
from classone.tokenizer import ClassOnePromptBuilder

HF_REPO_ID = "devops-thiago/classone-gemma4-e2b"


def compute_typesafe_confidence(probs: dict[str, float]) -> float:
    """Computes TypeSafe AI confidence metric: (N * P_max - 1) / (N - 1)."""
    n = len(probs)
    if n <= 1:
        return 1.0
    p_max = max(probs.values())
    conf = (n * p_max - 1.0) / (n - 1.0)
    return round(max(0.0, min(1.0, conf)), 4)


def load_model(device: str = "cuda:0"):
    print(f"[*] Loading published tokenizer and model from {HF_REPO_ID} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID)
    builder = ClassOnePromptBuilder(tokenizer)

    model = ClassOneModel.from_backbone(
        base_model_name_or_path=HF_REPO_ID,
        tokenizer=tokenizer,
        device=device,
        torch_dtype=torch.float16,
    )

    heads_path = hf_hub_download(HF_REPO_ID, "classone_heads.pt")
    heads = torch.load(heads_path, map_location=device)
    model.noul_head.load_state_dict(heads["noul_head"])
    model.choice_head.load_state_dict(heads["choice_head"])
    model.score_head.load_state_dict(heads["score_head"])
    model.eval()
    print("[*] Model loaded successfully!\n")
    return model, builder


def run_scenario(
    name: str,
    model: ClassOneModel,
    builder: ClassOnePromptBuilder,
    state: dict,
    questions: dict,
    device: str = "cuda:0",
):
    print("=" * 76)
    print(f"  SCENARIO: {name}")
    print("=" * 76)
    print("  State Input:")
    for k, v in state.items():
        print(f"    • {k}: {v}")
    print("\n  Questions Defined:")
    for q_id, q in questions.items():
        q_type = q.type if hasattr(q, "type") else type(q).__name__
        print(f"    • [{q_type.upper():<6}] {q_id}: {q.instructions}")

    packed = builder.pack(state=state, questions=questions)
    seq_len = packed.input_ids.shape[1]

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        results = model.evaluate_packed(packed)
    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    print(f"\n  Inference Latency: {latency_ms:.2f} ms | Prompt Sequence: {seq_len} tokens")
    print("  Results:")
    for q_id, res in results.items():
        if res.type == "noul":
            verdict = "YES (True)" if res.noul >= 0.5 else "NO (False)"
            print(f"    • {q_id} [Noul]: P(true) = {res.noul:.4f}  →  {verdict}")
        elif res.type == "choice":
            ts_conf = compute_typesafe_confidence(res.probabilities)
            print(
                f'    • {q_id} [Choice]: "{res.choice}" (Confidence: {res.confidence:.4f}, TypeSafe Conf: {ts_conf:.4f})'
            )
            print(f"        Probabilities: {res.probabilities}")
        elif res.type == "score":
            print(f"    • {q_id} [Score]: Expected Value = {res.score:.2f} / {len(res.probabilities)} levels")
            print(f"        Probabilities: {res.probabilities}")

    return {"name": name, "latency_ms": latency_ms, "seq_len": seq_len, "results": results}


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, builder = load_model(device=device)

    # 1. Multi-Facet Ticket Triage
    state_1 = {
        "customer": "Sarah Chen",
        "tier": "enterprise",
        "message": "We experienced an outage on our production webhook pipeline between 14:00 and 15:30 UTC. Several critical webhook deliveries failed. We need an explanation and SLA refund credit applied immediately.",
    }
    questions_1 = {
        "refund_request": NoulQuestion(instructions="Is the customer requesting a refund or billing credit?"),
        "team_routing": ChoiceQuestion(
            instructions="Target department for this ticket:",
            criteria={
                "billing": "Invoice disputes and billing credits",
                "infrastructure": "System outages, server downtime, and platform reliability",
                "security": "Vulnerabilities and auth breaches",
                "general": "Standard account questions",
            },
        ),
        "urgency": ScoreQuestion(
            instructions="Ticket urgency rating:",
            criteria=["low", "normal", "high", "critical_sla_breach"],
        ),
        "requires_escalation": NoulQuestion(instructions="Does this incident require senior manager escalation?"),
    }
    run_scenario("1. Multi-Facet Ticket Triage (Customer Support)", model, builder, state_1, questions_1, device)

    # 2. Agent Guardrails & Tool Verification ("Auto Mode")
    state_2 = {
        "agent_id": "code_refactor_agent",
        "action": "execute_shell_command",
        "command": "rm -rf /var/lib/docker/volumes/production_db_data && docker compose down -v",
        "target_environment": "production",
    }
    questions_2 = {
        "is_destructive": NoulQuestion(
            instructions="Does this proposed command destroy data or delete persistent volumes?"
        ),
        "safety_verdict": ChoiceQuestion(
            instructions="Security and safety verdict:",
            criteria={
                "auto_approve": "Safe read-only or low-risk operation",
                "prompt_user": "Action modifies files, requires user confirmation",
                "blocked": "Dangerous destructive or malicious operation",
            },
        ),
        "risk_score": ScoreQuestion(
            instructions="Action risk score:",
            criteria=["trivial", "moderate", "high_risk", "catastrophic"],
        ),
    }
    run_scenario("2. Agent Guardrails & Inline Safety Gating", model, builder, state_2, questions_2, device)

    # 3. Composite Scoring (Startup Pitch / Investment Memo)
    state_3 = {
        "company": "NeuroFast AI",
        "summary": "Building a single-pass System 1 decision model architecture for autonomous AI agents. Sub-50ms deterministic decisions with zero decoding overhead, replacing costly autoregressive LLM calls.",
        "market": "Global AI inference market ($45B TAM, growing at 32% CAGR).",
        "traction": "12 enterprise design partners, $80k ARR in 3 months.",
        "team": "Ex-Google DeepMind research leads.",
    }
    questions_3 = {
        "market_tam": ScoreQuestion(
            instructions="Market size and opportunity scale:",
            criteria=["niche", "medium_tam", "large_tam", "hyper_scale"],
        ),
        "defensibility": ScoreQuestion(
            instructions="Moat and technical defensibility:",
            criteria=["weak_wrapper", "commodity", "defensible_architecture", "breakthrough"],
        ),
        "execution_risk": NoulQuestion(instructions="Are there disqualifying team or execution red flags?"),
        "recommendation": ChoiceQuestion(
            instructions="Pipeline recommendation:",
            criteria={
                "pass": "Do not pursue",
                "diligence": "Schedule technical diligence",
                "partner_meeting": "Fast-track to partners",
            },
        ),
    }
    res_3 = run_scenario(
        "3. Composite Scoring (Investment Memo Evaluation)", model, builder, state_3, questions_3, device
    )
    # Composite formula demonstration: Weighted Score = 0.5 * Market + 0.5 * Defensibility
    m_score = res_3["results"]["market_tam"].score
    d_score = res_3["results"]["defensibility"].score
    composite_rating = 0.5 * m_score + 0.5 * d_score
    print(f"\n  [Code Composite Score]: 0.5 * {m_score:.2f} + 0.5 * {d_score:.2f} = {composite_rating:.2f} / 4.0")

    # 4. Intent Routing with Confidence-Gated Branching
    state_4 = {
        "user_query": "Can I upgrade my monthly subscription to the yearly pro tier with 20% discount?",
        "user_authenticated": True,
    }
    questions_4 = {
        "intent": ChoiceQuestion(
            instructions="Categorize user primary intent:",
            criteria={
                "upgrade_plan": "Subscription tier change or renewal",
                "technical_bug": "Product errors or downtime",
                "cancel_account": "Account termination",
                "other": "Unclassified query",
            },
        ),
        "has_discount_request": NoulQuestion(instructions="Is the user asking for a coupon, promo code, or discount?"),
        "readiness_score": ScoreQuestion(
            instructions="Purchase intent readiness:",
            criteria=["curious", "evaluating", "ready_to_buy"],
        ),
    }
    res_4 = run_scenario(
        "4. Intent Routing with Confidence-Gated Branching", model, builder, state_4, questions_4, device
    )
    # Confidence-gated routing in code
    choice_ans = res_4["results"]["intent"]
    conf = compute_typesafe_confidence(choice_ans.probabilities)
    if conf >= 0.85:
        route_decision = f"AUTO-EXECUTE branch '{choice_ans.choice}' (Confidence >= 0.85)"
    elif conf >= 0.55:
        route_decision = f"SOFT-APPLY branch '{choice_ans.choice}' with user confirmation"
    else:
        route_decision = "FALLBACK to human agent / LLM review (Low confidence)"
    print(f"\n  [Confidence-Gated Branch]: {route_decision}")

    # 5. RAG Retrieval Quality & Citation Verification
    state_5 = {
        "claim": "ClassOne achieves sub-50ms decision execution on edge GPUs without token generation.",
        "source_passage": "Measured on dual RTX 5060 Ti GPUs, ClassOne evaluated typed questions in 47.07 ms mean single-pass latency, producing typed probabilities without autoregressive token decoding.",
    }
    questions_5 = {
        "supports_claim": NoulQuestion(instructions="Does the passage factually support the stated claim?"),
        "citation_fidelity": ChoiceQuestion(
            instructions="Citation fidelity classification:",
            criteria={
                "direct_entailment": "Directly supports and confirms claim",
                "partial_overlap": "Partially mentions claim with gaps",
                "contradiction": "Contradicts claim",
            },
        ),
        "passage_relevance": ScoreQuestion(
            instructions="Passage relevance level:",
            criteria=["irrelevant", "somewhat_relevant", "highly_relevant"],
        ),
    }
    run_scenario("5. RAG Retrieval Quality & Citation Verification", model, builder, state_5, questions_5, device)

    print("\n" + "=" * 76)
    print("   ALL 5 TYPESAFE AI PATTERNS EVALUATED SUCCESSFULLY!")
    print("=" * 76)


if __name__ == "__main__":
    main()
