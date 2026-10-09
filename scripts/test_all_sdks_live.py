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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

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
    with ClassOneClient(base_url=base_url) as client:
        questions = {
            "intent": Choice(
                instructions="Select target support department:",
                criteria={
                    "billing": "Invoice disputes and duplicate refund requests",
                    "tech": "Application crash and service error",
                },
            ),
            "is_urgent": Noul(instructions="Is this an urgent priority issue?"),
            "risk_tier": Score(instructions="Assess customer churn risk:", criteria=["low", "medium", "critical"]),
        }
        # Connection & decision heads warmup
        client.decide(state="warmup", questions=questions)

        latencies = []
        res = None
        for _ in range(3):
            t0 = time.perf_counter()
            res = client.decide(
                state="Customer reports duplicate credit card charge of $49.00 on transaction #8832.",
                questions=questions,
            )
            latencies.append((time.perf_counter() - t0) * 1000.0)
        latencies.sort()
        latency_ms = latencies[1]

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
const questions = {{
  intent: new Choice("Select target support department:", {{
    billing: "Invoice disputes and duplicate refund requests",
    tech: "Application crash and service error"
  }}),
  is_urgent: new Noul("Is this an urgent priority issue?"),
  risk_tier: new Score("Assess customer churn risk:", ["low", "medium", "critical"])
}};

// Warmup connection and decision heads
await client.decide("warmup", questions);

const lats = [];
let res;
for (let i = 0; i < 3; i++) {{
  const t0 = performance.now();
  res = await client.decide(
    "Customer reports duplicate credit card charge of $49.00 on transaction #8832.",
    questions
  );
  lats.push(performance.now() - t0);
}}
lats.sort((a, b) => a - b);
const latency = lats[1];

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
    script = f"""package main

import (
\t"context"
\t"encoding/json"
\t"fmt"
\t"os"
\t"sort"
\t"time"

\t"github.com/devops-thiago/classone-sdks/go"
)

func main() {{
\tclient := classone.NewClient(classone.WithBaseURL("{base_url}"))
\treq := classone.NewDecision("Customer reports duplicate credit card charge of $49.00 on transaction #8832.").
\t\tWithChoice("intent", "Select target support department:", map[string]string{{
\t\t\t"billing": "Invoice disputes and duplicate refund requests",
\t\t\t"tech":    "Application crash and service error",
\t\t}}).
\t\tWithNoul("is_urgent", "Is this an urgent priority issue?").
\t\tWithScore("risk_tier", "Assess customer churn risk:", []string{{"low", "medium", "critical"}})

\t// Warmup HTTP connection
\tclient.Decide(context.Background(), req)

\tlats := make([]float64, 3)
\tvar res *classone.DecisionResponse
\tvar err error
\tfor i := 0; i < 3; i++ {{
\t\tt0 := time.Now()
\t\tres, err = client.Decide(context.Background(), req)
\t\tif err != nil {{
\t\t\tfmt.Fprintf(os.Stderr, "error: %v\\n", err)
\t\t\tos.Exit(1)
\t\t}}
\t\tlats[i] = float64(time.Since(t0).Microseconds()) / 1000.0
\t}}
\tsort.Float64s(lats)

\tpayload := map[string]any{{
\t\t"model":      res.Model,
\t\t"intent":     res.Choice("intent").Choice,
\t\t"noul":       res.Noul("is_urgent").Noul,
\t\t"score":      res.Score("risk_tier").Score,
\t\t"tokens":     res.Usage.InputTokens,
\t\t"latency_ms": lats[1],
\t}}
\tout, _ := json.Marshal(payload)
\tfmt.Println(string(out))
}}
"""
    bench_dir = go_dir / "bench"
    bench_dir.mkdir(exist_ok=True)
    bench_file = bench_dir / "main.go"
    bench_exe = go_dir / "bench_runner.exe"
    bench_file.write_text(script, encoding="utf-8")
    try:
        subprocess.check_call(["go", "build", "-o", str(bench_exe), "./bench/main.go"], cwd=go_dir)
        t0 = time.perf_counter()
        out = subprocess.check_output([str(bench_exe)], cwd=go_dir, text=True)
        wall_latency = (time.perf_counter() - t0) * 1000.0
        data = json.loads(out.strip())
        return {
            "language": "Go 1.26",
            "client": "classone.Client",
            "model": data["model"],
            "intent": data["intent"],
            "noul": round(data["noul"], 4),
            "score": round(data["score"], 2),
            "tokens": data["tokens"],
            "latency_ms": round(data.get("latency_ms", wall_latency), 2),
            "status": "PASS",
        }
    finally:
        if bench_file.exists():
            bench_file.unlink()
        if bench_exe.exists():
            bench_exe.unlink()
        if bench_dir.exists():
            try:
                bench_dir.rmdir()
            except OSError:
                pass


def test_rust_sdk(sdks_dir: Path, base_url: str) -> dict:
    rust_dir = sdks_dir / "rust"
    script = """use classone::ClassOneClient;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = ClassOneClient::new("__BASE_URL__", "api-key");

    // Warmup
    let _ = client
        .decide("warmup")
        .choice("intent", "dept", [("a", "a"), ("b", "b")])
        .noul("is_urgent", "urgent?")
        .score("risk_tier", "score", ["l", "m", "h"])
        .send();

    let mut lats = Vec::new();
    let mut last_res = None;
    for _ in 0..3 {
        let t0 = Instant::now();
        let res = client
            .decide("Customer reports duplicate credit card charge of $49.00 on transaction #8832.")
            .choice("intent", "Select target support department:", [
                ("billing", "Invoice disputes and duplicate refund requests"),
                ("tech", "Application crash and service error"),
            ])
            .noul("is_urgent", "Is this an urgent priority issue?")
            .score("risk_tier", "Assess customer churn risk:", ["low", "medium", "critical"])
            .send()?;
        lats.push(t0.elapsed().as_secs_f64() * 1000.0);
        last_res = Some(res);
    }
    lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let dt = lats[1];
    let res = last_res.unwrap();

    println!(
        r#"{{"model":"{}","intent":"{}","noul":{},"score":{},"tokens":{},"latency_ms":{:.2}}}"#,
        res.model,
        res.choice("intent").map(|c| c.choice.as_str()).unwrap_or(""),
        res.noul("is_urgent").map(|n| n.noul).unwrap_or(0.0),
        res.score("risk_tier").map(|s| s.score).unwrap_or(0.0),
        res.usage.input_tokens,
        dt
    );
    Ok(())
}
""".replace("__BASE_URL__", base_url)
    bench_file = rust_dir / "examples" / "bench.rs"
    bench_file.write_text(script, encoding="utf-8")
    rust_exe = rust_dir / "target" / "debug" / "examples" / "bench.exe"
    try:
        subprocess.check_call(["cargo", "build", "--example", "bench", "--quiet"], cwd=rust_dir)
        t0 = time.perf_counter()
        out = subprocess.check_output([str(rust_exe)], cwd=rust_dir, text=True)
        wall_latency = (time.perf_counter() - t0) * 1000.0
        data = json.loads(out.strip())
        return {
            "language": "Rust 1.98 (Cargo)",
            "client": "classone::ClassOneClient",
            "model": data["model"],
            "intent": data["intent"],
            "noul": round(data["noul"], 4),
            "score": round(data["score"], 2),
            "tokens": data["tokens"],
            "latency_ms": round(data.get("latency_ms", wall_latency), 2),
            "status": "PASS",
        }
    finally:
        if bench_file.exists():
            bench_file.unlink()


def test_java_sdk(sdks_dir: Path, base_url: str) -> dict:
    java_dir = sdks_dir / "java"
    bin_dir = java_dir / "bin"
    bin_dir.mkdir(exist_ok=True)

    bench_code = f"""import io.classone.*;
import java.util.List;
import java.util.Map;

public class BenchRunner {{
    public static void main(String[] args) throws Exception {{
        ClassOneClient client = ClassOneClient.builder()
                .baseUrl("{base_url}")
                .build();

        // Warmup connection
        try {{
            client.decide("warmup")
                    .choice("intent", "dept", Map.of("a", "a", "b", "b"))
                    .noul("is_urgent", "urgent?")
                    .score("risk_tier", "score", List.of("l", "m", "h"))
                    .execute();
        }} catch (Exception ignored) {{}}

        double[] lats = new double[3];
        ClassOneResponse res = null;
        for (int i = 0; i < 3; i++) {{
            long t0 = System.nanoTime();
            res = client.decide("Customer reports duplicate credit card charge of $49.00 on transaction #8832.")
                    .choice("intent", "Select target support department:", Map.of(
                            "billing", "Invoice disputes and duplicate refund requests",
                            "tech", "Application crash and service error"
                    ))
                    .noul("is_urgent", "Is this an urgent priority issue?")
                    .score("risk_tier", "Assess customer churn risk:", List.of("low", "medium", "critical"))
                    .execute();
            lats[i] = (System.nanoTime() - t0) / 1_000_000.0;
        }}
        java.util.Arrays.sort(lats);
        double latencyMs = lats[1];

        int inTok = res.usage() != null ? res.usage().getOrDefault("input_tokens", 0) : 0;
        System.out.printf(
            java.util.Locale.ROOT,
            "{{\\"model\\":\\"%s\\",\\"intent\\":\\"%s\\",\\"noul\\":%.4f,\\"score\\":%.2f,\\"tokens\\":%d,\\"latency_ms\\":%.2f}}%n",
            res.model(),
            res.selectedChoice("intent"),
            res.noulProbability("is_urgent"),
            res.rubricScore("risk_tier"),
            inTok,
            latencyMs
        );
    }}
}}
"""
    bench_file = java_dir / "BenchRunner.java"
    bench_file.write_text(bench_code, encoding="utf-8")
    try:
        java_files = list((java_dir / "src" / "main" / "java" / "io" / "classone").glob("*.java"))
        subprocess.check_call(
            ["javac", "-d", str(bin_dir)] + [str(f) for f in java_files] + [str(bench_file)],
            cwd=java_dir,
        )

        t0 = time.perf_counter()
        out = subprocess.check_output(["java", "-cp", str(bin_dir), "BenchRunner"], cwd=java_dir, text=True)
        wall_latency = (time.perf_counter() - t0) * 1000.0
        data = json.loads(out.strip())
        return {
            "language": "Java 21 (OpenJDK)",
            "client": "io.classone.ClassOneClient",
            "model": data["model"],
            "intent": data["intent"],
            "noul": round(data["noul"], 4),
            "score": round(data["score"], 2),
            "tokens": data["tokens"],
            "latency_ms": round(data.get("latency_ms", wall_latency), 2),
            "status": "PASS",
        }
    finally:
        if bench_file.exists():
            bench_file.unlink()


def test_ruby_sdk(sdks_dir: Path, base_url: str) -> dict:
    ruby_dir = sdks_dir / "ruby"
    script = f"""require_relative './lib/classone'
require 'json'

client = ClassOne::Client.new(base_url: '{base_url}')

# Warmup connection
client.decide('warmup') do |d|
  d.choice :intent, 'dept', {{ 'a' => 'a', 'b' => 'b' }}
  d.noul :is_urgent, 'urgent?'
  d.score :risk_tier, 'score', %w[l m h]
end

lats = []
res = nil
3.times do
  t0 = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  res = client.decide('Customer reports duplicate credit card charge of $49.00 on transaction #8832.') do |d|
    d.choice :intent, 'Select target support department:', {{
      'billing' => 'Invoice disputes and duplicate refund requests',
      'tech'    => 'Application crash and service error'
    }}
    d.noul :is_urgent, 'Is this an urgent priority issue?'
    d.score :risk_tier, 'Assess customer churn risk:', %w[low medium critical]
  end
  lats << (Process.clock_gettime(Process::CLOCK_MONOTONIC) - t0) * 1000.0
end
lats.sort!
dt_ms = lats[1]

in_tok = res.usage['input_tokens'] || 0
puts JSON.generate({{
  model: res.model,
  intent: res.choice(:intent).selected,
  noul: res.noul(:is_urgent).probability,
  score: res.score(:risk_tier).value,
  tokens: in_tok,
  latency_ms: dt_ms
}})
"""
    bench_file = ruby_dir / "_bench.rb"
    bench_file.write_text(script, encoding="utf-8")
    try:
        t0 = time.perf_counter()
        out = subprocess.check_output(["ruby", str(bench_file)], cwd=ruby_dir, text=True)
        wall_latency = (time.perf_counter() - t0) * 1000.0
        data = json.loads(out.strip())
        return {
            "language": "Ruby 3.3",
            "client": "ClassOne::Client",
            "model": data["model"],
            "intent": data["intent"],
            "noul": round(data["noul"], 4),
            "score": round(data["score"], 2),
            "tokens": data["tokens"],
            "latency_ms": round(data.get("latency_ms", wall_latency), 2),
            "status": "PASS",
        }
    finally:
        if bench_file.exists():
            bench_file.unlink()


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
