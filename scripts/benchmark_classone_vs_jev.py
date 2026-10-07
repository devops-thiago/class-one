#!/usr/bin/env python3
"""Head-to-Head Benchmark: ClassOne (Local RTX 5060 Ti) vs TypeSafe AI Jev (Cloud API).

Compares:
1. Decision accuracy, probability distributions, and confidence values across 5 canonical scenarios.
2. Latency percentiles (Mean, P50, P90, P95, Min, Max) and throughput (req/s).
"""

import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from classone.modeling.modeling_classone import ClassOneModel
from classone.schemas import ChoiceQuestion, NoulQuestion, ScoreQuestion
from classone.tokenizer import ClassOnePromptBuilder

# Route api.typesafe.ai to 104.18.26.46 to bypass ISP null-route on 104.18.24.46
_old_getaddrinfo = socket.getaddrinfo


def _custom_getaddrinfo(host, port, *args, **kwargs):
    if host == "api.typesafe.ai":
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("104.18.26.46", port))]
    return _old_getaddrinfo(host, port, *args, **kwargs)


socket.getaddrinfo = _custom_getaddrinfo

HF_REPO_ID = "devops-thiago/classone-gemma4-e2b"
JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"


def get_jev_token() -> str | None:
    token = os.environ.get("JEV_TOKEN") or os.environ.get("TYPESAFE_API_KEY")
    if token:
        return token
    if os.path.exists(".env"):
        with open(".env", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("JEV_TOKEN=") or line.startswith("TYPESAFE_API_KEY="):
                    return line.split("=", 1)[1].strip().strip("'\"")
    return None


def call_jev_api(state: Any, questions: dict, token: str) -> tuple[dict, float]:
    """Sends decision request to TypeSafe AI Jev API and measures round-trip latency."""
    q_payload = {}
    for q_id, q in questions.items():
        if isinstance(q, NoulQuestion):
            q_payload[q_id] = {"type": "noul", "instructions": q.instructions}
        elif isinstance(q, ChoiceQuestion):
            q_payload[q_id] = {"type": "choice", "instructions": q.instructions, "criteria": q.criteria}
        elif isinstance(q, ScoreQuestion):
            q_payload[q_id] = {"type": "score", "instructions": q.instructions, "criteria": q.criteria}

    req_data = {
        "model": "jev-latest",
        "state": state,
        "questions": q_payload,
    }

    req = urllib.request.Request(
        JEV_ENDPOINT,
        data=json.dumps(req_data).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
            "User-Agent": "classone-benchmark/0.1.0",
        },
        method="POST",
    )

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return body, latency_ms


def load_classone_model(device: str = "cuda:0"):
    print(f"[*] Loading ClassOne model from {HF_REPO_ID} on {device}...")
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
    print("[✓] ClassOne model ready.\n")
    return model, builder


def evaluate_classone(
    model: ClassOneModel, builder: ClassOnePromptBuilder, state: Any, questions: dict, device_idx: int = 0
) -> tuple[dict, float]:
    """Runs single-pass evaluation on ClassOne model with CUDA synchronization."""
    packed = builder.pack(state=state, questions=questions)
    torch.cuda.synchronize(device_idx)
    t0 = time.perf_counter()
    with torch.no_grad():
        answers = model.evaluate_packed(packed)
    torch.cuda.synchronize(device_idx)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return answers, latency_ms


def format_ms(val: float) -> str:
    return f"{val:.2f} ms"


def main():
    jev_token = get_jev_token()
    if not jev_token:
        print("[!] Error: JEV_TOKEN or TYPESAFE_API_KEY not found in .env or environment.", file=sys.stderr)
        sys.exit(1)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_idx = 0 if torch.cuda.is_available() else None
    co_model, co_builder = load_classone_model(device=device)

    # Scenarios to benchmark
    scenarios = [
        {
            "id": "support_triage",
            "name": "1. Multi-Facet Support Ticket Triage",
            "state": {
                "customer": "Sarah Chen",
                "tier": "enterprise",
                "message": "We experienced an outage on our production webhook pipeline between 14:00 and 15:30 UTC. Several critical webhook deliveries failed. We need an explanation and SLA refund credit applied immediately.",
            },
            "questions": {
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
            },
        },
        {
            "id": "agent_guardrail",
            "name": "2. Agent Guardrails & Tool Gating",
            "state": {
                "agent_id": "code_refactor_agent",
                "action": "execute_shell_command",
                "command": "rm -rf /var/lib/docker/volumes/production_db_data && docker compose down -v",
                "target_environment": "production",
            },
            "questions": {
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
            },
        },
        {
            "id": "composite_score",
            "name": "3. Composite Scoring (Investment Memo)",
            "state": {
                "company": "NeuroFast AI",
                "summary": "Building a single-pass System 1 decision model architecture for autonomous AI agents. Sub-50ms deterministic decisions with zero decoding overhead, replacing costly autoregressive LLM calls.",
                "market": "Global AI inference market ($45B TAM, growing at 32% CAGR).",
                "traction": "12 enterprise design partners, $80k ARR in 3 months.",
                "team": "Ex-Google DeepMind research leads.",
            },
            "questions": {
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
            },
        },
        {
            "id": "intent_routing",
            "name": "4. Intent Routing & Readiness",
            "state": {
                "user_query": "Can I upgrade my monthly subscription to the yearly pro tier with 20% discount?",
                "user_authenticated": True,
            },
            "questions": {
                "intent": ChoiceQuestion(
                    instructions="Categorize user primary intent:",
                    criteria={
                        "upgrade_plan": "Subscription tier change or renewal",
                        "technical_bug": "Product errors or downtime",
                        "cancel_account": "Account termination",
                        "other": "Unclassified query",
                    },
                ),
                "has_discount_request": NoulQuestion(
                    instructions="Is the user asking for a coupon, promo code, or discount?"
                ),
                "readiness_score": ScoreQuestion(
                    instructions="Purchase intent readiness:",
                    criteria=["curious", "evaluating", "ready_to_buy"],
                ),
            },
        },
        {
            "id": "rag_verification",
            "name": "5. RAG Retrieval & Citation Check",
            "state": {
                "claim": "ClassOne achieves sub-50ms decision execution on edge GPUs without token generation.",
                "source_passage": "Measured on dual RTX 5060 Ti GPUs, ClassOne evaluated typed questions in 47.07 ms mean single-pass latency, producing typed probabilities without autoregressive token decoding.",
            },
            "questions": {
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
            },
        },
    ]

    print("=" * 80)
    print("      PART 1: SIDE-BY-SIDE DECISION COMPARISON ACROSS 5 SCENARIOS")
    print("=" * 80)

    for sc in scenarios:
        print(f"\n>>> {sc['name']}")
        co_res, co_lat = evaluate_classone(co_model, co_builder, sc["state"], sc["questions"], device_idx)
        jev_res, jev_lat = call_jev_api(sc["state"], sc["questions"], jev_token)

        jev_answers = jev_res.get("answers", {})

        print(f"    Latency — ClassOne (Local): {co_lat:.2f} ms  |  TypeSafe Jev (Cloud): {jev_lat:.2f} ms")
        for q_id in sc["questions"].keys():
            # ClassOne answer
            c_ans = co_res[q_id]
            j_ans = jev_answers.get(q_id, {})
            q_type = c_ans.type

            if q_type == "noul":
                c_val = f"P={c_ans.noul:.2f}"
                j_val = f"P={j_ans.get('noul', 0.0):.2f}"
                match = "AGREED" if (c_ans.noul >= 0.5) == (j_ans.get("noul", 0.0) >= 0.5) else "DIVERGED"
                print(f"      • {q_id:<20} [NOUL]   ClassOne: {c_val:<10} │ Jev: {j_val:<10}  ({match})")
            elif q_type == "choice":
                c_val = f"'{c_ans.choice}' ({c_ans.confidence:.2f})"
                j_val = f"'{j_ans.get('choice', '')}' ({j_ans.get('confidence', 0.0):.2f})"
                match = "AGREED" if c_ans.choice == j_ans.get("choice") else "DIVERGED"
                print(f"      • {q_id:<20} [CHOICE] ClassOne: {c_val:<22} │ Jev: {j_val:<22}  ({match})")
            elif q_type == "score":
                c_val = f"Score={c_ans.score:.2f}"
                j_score = j_ans.get("score", 0.0)
                # Jev 0-indexed scores vs ClassOne 1-indexed scores (normalize display)
                j_val = f"Score={j_score:.2f} (idx)"
                print(f"      • {q_id:<20} [SCORE]  ClassOne: {c_val:<18} │ Jev: {j_val:<18}")

    print("\n" + "=" * 80)
    print("      PART 2: LATENCY & THROUGHPUT BENCHMARK (15 RUNS PER SYSTEM)")
    print("=" * 80)

    bench_scenario = scenarios[0]  # Multi-facet support triage (3 typed questions)
    n_iterations = 15

    # Warmup ClassOne
    for _ in range(3):
        evaluate_classone(co_model, co_builder, bench_scenario["state"], bench_scenario["questions"], device_idx)

    print(f"[*] Benchmarking ClassOne (Local RTX 5060 Ti) over {n_iterations} runs...")
    co_times = []
    for _ in range(n_iterations):
        _, lat = evaluate_classone(
            co_model, co_builder, bench_scenario["state"], bench_scenario["questions"], device_idx
        )
        co_times.append(lat)

    print(f"[*] Benchmarking TypeSafe Jev (Cloud API) over {n_iterations} runs...")
    jev_times = []
    for i in range(n_iterations):
        _, lat = call_jev_api(bench_scenario["state"], bench_scenario["questions"], jev_token)
        jev_times.append(lat)
        time.sleep(0.1)  # polite spacing between cloud requests

    co_mean = float(np.mean(co_times))
    co_p50 = float(np.percentile(co_times, 50))
    co_p90 = float(np.percentile(co_times, 90))
    co_p95 = float(np.percentile(co_times, 95))
    co_min = float(np.min(co_times))
    co_max = float(np.max(co_times))
    co_rps = 1000.0 / co_mean

    jev_mean = float(np.mean(jev_times))
    jev_p50 = float(np.percentile(jev_times, 50))
    jev_p90 = float(np.percentile(jev_times, 90))
    jev_p95 = float(np.percentile(jev_times, 95))
    jev_min = float(np.min(jev_times))
    jev_max = float(np.max(jev_times))
    jev_rps = 1000.0 / jev_mean

    speedup = jev_mean / co_mean

    col1 = 26
    col2 = 25
    col3 = 25
    sep = f"{'─' * col1}┼{'─' * (col2 + 2)}┼{'─' * (col3 + 2)}"

    print("\n" + "=" * 80)
    print("             HEAD-TO-HEAD LATENCY BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Metric':<{col1}} │ {'ClassOne (Local RTX 5060 Ti)':>{col2}} │ {'TypeSafe Jev (Cloud API)':>{col3}}")
    print(sep)
    print(f"{'Mean Latency':<{col1}} │ {format_ms(co_mean):>{col2}} │ {format_ms(jev_mean):>{col3}}")
    print(f"{'Median (P50)':<{col1}} │ {format_ms(co_p50):>{col2}} │ {format_ms(jev_p50):>{col3}}")
    print(f"{'P90 Latency':<{col1}} │ {format_ms(co_p90):>{col2}} │ {format_ms(jev_p90):>{col3}}")
    print(f"{'P95 Latency':<{col1}} │ {format_ms(co_p95):>{col2}} │ {format_ms(jev_p95):>{col3}}")
    print(f"{'Min Latency':<{col1}} │ {format_ms(co_min):>{col2}} │ {format_ms(jev_min):>{col3}}")
    print(f"{'Max Latency':<{col1}} │ {format_ms(co_max):>{col2}} │ {format_ms(jev_max):>{col3}}")
    print(f"{'Throughput':<{col1}} │ {f'{co_rps:.1f} req/s':>{col2}} │ {f'{jev_rps:.1f} req/s':>{col3}}")
    print(f"{'Deployment':<{col1}} │ {'Local Edge GPU':>{col2}} │ {'Cloud API':>{col3}}")
    print(f"{'Cost per decision':<{col1}} │ {'$0.00 (Self-hosted)':>{col2}} │ {'Cloud billed':>{col3}}")
    print(sep)
    print(f"  ⚡  Speedup: ClassOne is {speedup:.1f}x FASTER than cloud Jev API round-trip")
    print("=" * 80 + "\n")

    results_out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runs": n_iterations,
        "classone": {
            "model": HF_REPO_ID,
            "device": "NVIDIA GeForce RTX 5060 Ti",
            "mean_ms": round(co_mean, 2),
            "p50_ms": round(co_p50, 2),
            "p90_ms": round(co_p90, 2),
            "p95_ms": round(co_p95, 2),
            "min_ms": round(co_min, 2),
            "max_ms": round(co_max, 2),
            "throughput_rps": round(co_rps, 2),
        },
        "typesafe_jev": {
            "model": "jev-1.13.0",
            "endpoint": JEV_ENDPOINT,
            "mean_ms": round(jev_mean, 2),
            "p50_ms": round(jev_p50, 2),
            "p90_ms": round(jev_p90, 2),
            "p95_ms": round(jev_p95, 2),
            "min_ms": round(jev_min, 2),
            "max_ms": round(jev_max, 2),
            "throughput_rps": round(jev_rps, 2),
        },
        "speedup_factor": round(speedup, 2),
    }

    out_path = os.path.join("benchmarks", "results", "benchmark_classone_vs_jev.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results_out, f, indent=2)
    print(f"[✓] Full benchmark data exported to: {out_path}")


if __name__ == "__main__":
    main()
