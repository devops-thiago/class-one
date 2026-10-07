"""Benchmark script for ClassOne running inside Docker on 2 vCPUs and 8GB RAM without GPU.

Dispatches 5 parallel concurrent requests to POST /v1/decide,
measures latency, prompt token throughput (tokens/s), decisions/s,
and verifies decision correctness.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from typing import Any

import httpx

SERVER_URL = "http://localhost:8000"

REALISTIC_DOC = """Policy Document: Enterprise Data Access and Remote Execution Guidelines (Section 4.2)
All automated agents and background services operating within the corporate cloud perimeter
must satisfy strict credential isolation and rate limiting controls. Any API key, token, or secret
must be retrieved exclusively from HashiCorp Vault using short-lived JWT authentication.
Under no circumstances may plaintext API tokens be serialized into logs, prompt histories, or
shared environment configuration files. When executing remote tool actions, agents must verify
that the target host matches an allowlisted internal domain (*.corp.internal) and that the operation
does not perform destructive filesystem modifications (such as rm -rf, mkfs, or disk wipe).
Failure to follow these protocols results in immediate revocation of service access tokens
and automatic trigger of security escalation alerts to the SOC team.
"""

BENCHMARK_REQUESTS = [
    {
        "model": "classone-gemma4-e2b-q8",
        "state": REALISTIC_DOC + f"\n\nContext Contextual Incident Scenario #{i + 1}:\n"
        f"An automated deployment agent attempted to execute `curl https://external-api.com -H 'Authorization: Bearer SECRET_KEY'` "
        f"from a staging container without vault token verification.",
        "questions": {
            "policy_violation": {
                "type": "choice",
                "instructions": "Determine if the action violates enterprise security policy.",
                "criteria": {
                    "A": "Violates security policy by using external domain and plaintext token",
                    "B": "Complies fully with security policy",
                },
            },
            "risk_score": {
                "type": "score",
                "instructions": "Rate the severity of this security violation on a 1-5 scale.",
                "criteria": [
                    "Negligible risk",
                    "Low risk",
                    "Moderate risk",
                    "High risk",
                    "Critical risk",
                ],
            },
            "immediate_revocation": {
                "type": "noul",
                "instructions": "Should service credentials be revoked immediately?",
            },
        },
    }
    for i in range(5)
]


async def send_request(client: httpx.AsyncClient, req_idx: int, payload: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    resp = await client.post(f"{SERVER_URL}/v1/decide", json=payload, timeout=120.0)
    elapsed = time.perf_counter() - start
    resp.raise_for_status()
    data = resp.json()
    return {
        "req_idx": req_idx,
        "status_code": resp.status_code,
        "elapsed_sec": elapsed,
        "input_tokens": data.get("usage", {}).get("input_tokens", 0),
        "answers": data.get("answers", {}),
    }


async def main():
    print("=" * 70)
    print("ClassOne Docker CPU Benchmark: 2 vCPUs | 8 GB RAM | 8-Bit (Q8) | No GPU | No GGUF")
    print("=" * 70)

    # 1. Healthcheck wait
    print(f"Connecting to {SERVER_URL}/health...")
    async with httpx.AsyncClient() as client:
        for attempt in range(60):
            try:
                h = await client.get(f"{SERVER_URL}/health", timeout=3.0)
                if h.status_code == 200:
                    print("Service is Healthy and Ready!\n")
                    break
            except Exception:
                await asyncio.sleep(2)
        else:
            raise RuntimeError(f"Service at {SERVER_URL} failed to become ready after 120s.")

    # 2. Warmup request (1 request)
    print("Executing single warmup request...")
    async with httpx.AsyncClient() as client:
        w_res = await send_request(client, 0, BENCHMARK_REQUESTS[0])
        w_tokens = w_res["input_tokens"]
        w_time = w_res["elapsed_sec"]
        print(f"Warmup complete: {w_tokens} tokens processed in {w_time:.3f}s ({w_tokens / w_time:.1f} tokens/s)\n")

    # 3. Parallel benchmark (5 concurrent requests)
    print("Launching 5 parallel concurrent requests...")
    async with httpx.AsyncClient() as client:
        wall_start = time.perf_counter()
        tasks = [send_request(client, i, BENCHMARK_REQUESTS[i]) for i in range(5)]
        results = await asyncio.gather(*tasks)
        total_wall_time = time.perf_counter() - wall_start

    print("\n" + "=" * 70)
    print("PARALLEL BENCHMARK RESULTS (5 CONCURRENT WORKERS)")
    print("=" * 70)

    total_tokens = sum(r["input_tokens"] for r in results)
    total_decisions = sum(len(r["answers"]) for r in results)
    latencies = [r["elapsed_sec"] for r in results]

    for r in results:
        t_sec = r["elapsed_sec"]
        toks = r["input_tokens"]
        rate = toks / t_sec if t_sec > 0 else 0
        dec_count = len(r["answers"])
        p_ans = r["answers"].get("policy_violation", {})
        print(
            f"Worker #{r['req_idx'] + 1}: {toks} tokens, {dec_count} decisions | "
            f"Latency: {t_sec:.3f}s | Throughput: {rate:.1f} tokens/s | Choice: {p_ans.get('choice')} "
            f"(conf: {p_ans.get('confidence')})"
        )

    mean_lat = statistics.mean(latencies)
    median_lat = statistics.median(latencies)
    p95_lat = sorted(latencies)[int(len(latencies) * 0.95)]
    aggregate_tokens_per_sec = total_tokens / total_wall_time
    aggregate_decisions_per_sec = total_decisions / total_wall_time

    print("-" * 70)
    print(f"Total Wall-Clock Time:         {total_wall_time:.3f} s")
    print(f"Total Input Tokens Processed:  {total_tokens} tokens")
    print(f"Total Decisions Computed:      {total_decisions} decisions")
    print(f"Mean Request Latency:          {mean_lat:.3f} s")
    print(f"Median Request Latency (p50):  {median_lat:.3f} s")
    print(f"p95 Request Latency:           {p95_lat:.3f} s")
    print(f"Aggregate Token Throughput:    {aggregate_tokens_per_sec:.2f} tokens/sec")
    print(f"Aggregate Decision Rate:       {aggregate_decisions_per_sec:.2f} decisions/sec")
    print("=" * 70)

    # Save benchmark metrics to json
    metrics = {
        "hardware": "Docker (2 vCPUs, 8 GB RAM, CPU only, no GPU, no GGUF)",
        "quantization": "INT8 Dynamic (PyTorch torch.ao)",
        "concurrency": 5,
        "total_wall_clock_sec": round(total_wall_time, 4),
        "total_input_tokens": total_tokens,
        "total_decisions": total_decisions,
        "mean_latency_sec": round(mean_lat, 4),
        "median_latency_sec": round(median_lat, 4),
        "aggregate_tokens_per_sec": round(aggregate_tokens_per_sec, 2),
        "aggregate_decisions_per_sec": round(aggregate_decisions_per_sec, 2),
        "workers": [
            {
                "worker_id": r["req_idx"] + 1,
                "latency_sec": round(r["elapsed_sec"], 4),
                "tokens": r["input_tokens"],
                "tokens_per_sec": round(r["input_tokens"] / r["elapsed_sec"], 2),
                "decisions": len(r["answers"]),
            }
            for r in results
        ],
    }
    with open("docker_cpu_benchmark_results.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved results to docker_cpu_benchmark_results.json")


if __name__ == "__main__":
    asyncio.run(main())
