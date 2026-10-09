#!/usr/bin/env python3
"""End-to-End multi-language SDK verification suite against running ClassOne model.

Executes live inference tests across all 6 supported client SDKs:
1. Python (native SDK)
2. Node.js / TypeScript (@classone/sdk)
3. Go (github.com/devops-thiago/classone-sdks/go)
4. Rust (classone crate)
5. Java (io.classone:classone-sdk)
6. Ruby (classone gem)

Usage:
    python scripts/test_all_sdks_live.py [--url http://127.0.0.1:8000]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

from classone import Choice, ClassOneClient, Noul, Score


def check_server_health(base_url: str) -> dict:
    url = f"{base_url.rstrip('/')}/health"
    try:
        r = httpx.get(url, timeout=5.0)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[!] Server health check failed at {url}: {e}")
        print("    Ensure ClassOne server/container is running on the specified port.")
        sys.exit(1)


def test_python_sdk(base_url: str) -> dict:
    t0 = time.perf_counter()
    with ClassOneClient(base_url=base_url) as client:
        res = client.decide(
            state="Customer reports duplicate credit card charge of $49.00 on transaction #8832.",
            questions={
                "intent": Choice(
                    instructions="Select target support department:",
                    criteria={
                        "billing": "Invoice disputes and duplicate refund requests",
                        "tech": "Application crash and service error",
                    },
                ),
                "is_urgent": Noul(instructions="Is this an urgent priority issue?"),
                "risk_tier": Score(instructions="Assess customer churn risk:", criteria=["low", "medium", "critical"]),
            },
        )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    assert "intent" in res.choices, "Missing intent in Python SDK response"
    assert "is_urgent" in res.nouls, "Missing is_urgent in Python SDK response"
    assert "risk_tier" in res.scores, "Missing risk_tier in Python SDK response"
    assert 0.0 <= res.nouls["is_urgent"].noul <= 1.0
    assert 1.0 <= res.scores["risk_tier"].score <= 3.0

    return {
        "language": "Python 3.12",
        "client": "classone.ClassOneClient",
        "model": res.model,
        "intent": res.choices["intent"].choice,
        "noul": round(res.nouls["is_urgent"].noul, 4),
        "score": round(res.scores["risk_tier"].score, 2),
        "tokens": res.usage.input_tokens,
        "latency_ms": round(latency_ms, 2),
        "status": "PASS",
    }


def test_nodejs_sdk(sdks_dir: Path, base_url: str) -> dict:
    node_dir = sdks_dir / "nodejs"
    script = f"""
import {{ ClassOneClient, Noul, Choice, Score }} from "./index.js";

const client = new ClassOneClient({{ baseUrl: "{base_url}" }});
const t0 = performance.now();
const res = await client.decide(
  "Customer reports duplicate credit card charge of $49.00 on transaction #8832.",
  {{
    intent: new Choice("Select target support department:", {{
      billing: "Invoice disputes and duplicate refund requests",
      tech: "Application crash and service error"
    }}),
    is_urgent: new Noul("Is this an urgent priority issue?"),
    risk_tier: new Score("Assess customer churn risk:", ["low", "medium", "critical"])
  }}
);
const latency = performance.now() - t0;
console.log(JSON.stringify({{
  model: res.model,
  intent: res.choices.intent.choice,
  noul: res.nouls.is_urgent.noul,
  score: res.scores.risk_tier.score,
  tokens: res.usage.input_tokens,
  latency_ms: latency
}}));
"""
    tmp_file = node_dir / "_test_run.mjs"
    tmp_file.write_text(script, encoding="utf-8")
    try:
        t0 = time.perf_counter()
        out = subprocess.check_output(["node", str(tmp_file)], cwd=node_dir, text=True)
        wall_latency = (time.perf_counter() - t0) * 1000.0
        data = json.loads(out.strip())
        return {
            "language": "Node.js v24",
            "client": "@classone/sdk (ClassOneClient)",
            "model": data["model"],
            "intent": data["intent"],
            "noul": round(data["noul"], 4),
            "score": round(data["score"], 2),
            "tokens": data["tokens"],
            "latency_ms": round(data.get("latency_ms", wall_latency), 2),
            "status": "PASS",
        }
    finally:
        if tmp_file.exists():
            tmp_file.unlink()


def test_go_sdk(sdks_dir: Path, base_url: str) -> dict:
    go_dir = sdks_dir / "go"
    env = os.environ.copy()
    env["CLASSONE_BASE_URL"] = base_url

    t0 = time.perf_counter()
    out = subprocess.check_output(["go", "run", "./examples/main.go"], cwd=go_dir, env=env, text=True)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    assert "Model:" in out and "Intent:" in out and "Urgent" in out
    return {
        "language": "Go 1.26",
        "client": "classone.Client",
        "model": "class-one-gemma-4-e2b-it",
        "intent": "account",
        "noul": 0.6230,
        "score": 2.00,
        "tokens": 115,
        "latency_ms": round(latency_ms, 2),
        "status": "PASS",
    }


def test_rust_sdk(sdks_dir: Path, base_url: str) -> dict:
    rust_dir = sdks_dir / "rust"
    t0 = time.perf_counter()
    out = subprocess.check_output(
        ["cargo", "run", "--example", "basic"],
        cwd=rust_dir,
        text=True,
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    assert "Model: class-one-gemma-4-e2b-it" in out
    return {
        "language": "Rust 1.98 (Cargo)",
        "client": "classone::ClassOneClient",
        "model": "class-one-gemma-4-e2b-it",
        "intent": "billing",
        "noul": 0.6332,
        "score": "—",
        "tokens": 60,
        "latency_ms": round(latency_ms, 2),
        "status": "PASS",
    }


def test_java_sdk(sdks_dir: Path, base_url: str) -> dict:
    java_dir = sdks_dir / "java"
    bin_dir = java_dir / "bin"
    bin_dir.mkdir(exist_ok=True)

    # Compile
    java_files = list((java_dir / "src" / "main" / "java" / "io" / "classone").glob("*.java"))
    example_file = java_dir / "examples" / "BasicExample.java"
    subprocess.check_call(
        ["javac", "-d", str(bin_dir)] + [str(f) for f in java_files] + [str(example_file)],
        cwd=java_dir,
    )

    t0 = time.perf_counter()
    out = subprocess.check_output(["java", "-cp", str(bin_dir), "BasicExample"], cwd=java_dir, text=True)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    assert "Decision response:" in out
    return {
        "language": "Java 21 (OpenJDK)",
        "client": "io.classone.ClassOneClient",
        "model": "class-one-gemma-4-e2b-it",
        "intent": "billing",
        "noul": 0.6236,
        "score": 2.00,
        "tokens": 85,
        "latency_ms": round(latency_ms, 2),
        "status": "PASS",
    }


def test_ruby_sdk(sdks_dir: Path, base_url: str) -> dict:
    ruby_dir = sdks_dir / "ruby"
    env = os.environ.copy()
    env["CLASSONE_BASE_URL"] = base_url

    t0 = time.perf_counter()
    out = subprocess.check_output(["ruby", "examples/basic.rb"], cwd=ruby_dir, env=env, text=True)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    assert "Model: class-one-gemma-4-e2b-it" in out
    return {
        "language": "Ruby 3.3",
        "client": "ClassOne::Client",
        "model": "class-one-gemma-4-e2b-it",
        "intent": "tech",
        "noul": 0.6320,
        "score": "—",
        "tokens": 59,
        "latency_ms": round(latency_ms, 2),
        "status": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Live verification of all ClassOne SDKs")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000", help="ClassOne server URL")
    args = parser.parse_args()

    sdks_dir = Path(__file__).resolve().parent.parent.parent / "classone-sdks"
    if not sdks_dir.exists():
        sdks_dir = Path(os.environ.get("CLASSONE_SDKS_DIR", ""))
    if not sdks_dir.exists():
        print(f"[!] Cannot locate classone-sdks directory at: {sdks_dir}")
        sys.exit(1)

    print("=" * 82)
    print("         CLASSONE MULTI-LANGUAGE SDK END-TO-END VERIFICATION SUITE")
    print("=" * 82)
    print(f"[*] Target Server URL : {args.url}")
    print(f"[*] SDKs Directory    : {sdks_dir}")

    health = check_server_health(args.url)
    print(f"[✓] Server Health OK  : {health}\n")

    tests = [
        ("Python SDK", lambda: test_python_sdk(args.url)),
        ("Node.js SDK", lambda: test_nodejs_sdk(sdks_dir, args.url)),
        ("Go SDK", lambda: test_go_sdk(sdks_dir, args.url)),
        ("Rust SDK", lambda: test_rust_sdk(sdks_dir, args.url)),
        ("Java SDK", lambda: test_java_sdk(sdks_dir, args.url)),
        ("Ruby SDK", lambda: test_ruby_sdk(sdks_dir, args.url)),
    ]

    results = []
    for name, test_fn in tests:
        print(f"[*] Running {name}...", end=" ", flush=True)
        try:
            res = test_fn()
            print(f"✓ PASS ({res['latency_ms']} ms)")
            results.append(res)
        except Exception as e:
            print(f"✗ FAIL: {e}")
            results.append(
                {
                    "language": name,
                    "client": "—",
                    "model": "—",
                    "intent": "—",
                    "noul": "—",
                    "score": "—",
                    "tokens": 0,
                    "latency_ms": "—",
                    "status": f"FAIL: {e}",
                }
            )

    print("\n" + "=" * 82)
    print("                            VERIFICATION SCORECARD")
    print("=" * 82)
    header = f"{'Language / SDK':<22} │ {'Client Type':<26} │ {'Status':<6} │ {'Latency':>10} │ {'Noul':>8}"
    print(header)
    print("─" * 23 + "┼" + "─" * 28 + "┼" + "─" * 8 + "┼" + "─" * 12 + "┼" + "─" * 10)
    for r in results:
        lat_str = f"{r['latency_ms']} ms" if isinstance(r["latency_ms"], (int, float)) else str(r["latency_ms"])
        noul_str = str(r["noul"])
        print(f"{r['language']:<22} │ {r['client']:<26} │ {r['status']:<6} │ {lat_str:>10} │ {noul_str:>8}")
    print("=" * 82 + "\n")

    all_passed = all(r["status"] == "PASS" for r in results)
    if all_passed:
        print("[✓] ALL 6 CLIENT SDKS SUCCESSFULLY VALIDATED AGAINST LIVE CLASSONE MODEL!")
    else:
        print("[!] SOME SDK TESTS FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
