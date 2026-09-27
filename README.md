# ClassOne: System 1 Decision Model Architecture

[![CI](https://github.com/devops-thiago/class-one/actions/workflows/ci.yml/badge.svg)](https://github.com/devops-thiago/class-one/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**ClassOne** is an open-source **System 1 decision model architecture** built on Google's **Gemma 4 E2B** (Effective 2B). It provides rapid, non-autoregressive, schema-constrained decision execution for classification, routing, verification, and scoring in a single forward pass.

Inspired by Daniel Kahneman's cognitive framework in *Thinking, Fast and Slow* and modern proper scoring rules (Brier, 1950; Gneiting & Raftery, 2007).

---

## Key Features

- **System 1 Architecture:** Non-autoregressive decision model. Skips open-ended token generation entirely.
- **Single-Pass Parallel Evaluation:** Evaluates multiple typed questions (`Noul`, `Choice`, `Score`) simultaneously in a single forward pass over unstructured state (<10ms on MPS, <50ms on edge GPU).
- **Zero Structural Hallucinations:** Constrained by design to valid schema outputs.
- **Calibrated Probabilities (RLCD):** Trained and scored using Strictly Proper Scoring Rules (normalized Brier score + Negative Log-Likelihood) to penalize overconfidence.
- **Dual API Endpoints:** Exposes `POST /v1/decide` and `POST /v1/classone` for schema-out decision queries.
- **Python Client SDK:** Idiomatic synchronous and asynchronous client library (`from classone import ClassOneClient`).

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/devops-thiago/class-one.git
cd class-one
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Run Test Suite

```bash
pytest tests/
```

### 3. Start the API Server

```bash
uvicorn classone.server.app:app --host 0.0.0.0 --port 8000
```

### 4. Query via HTTP

```bash
curl -X POST http://localhost:8000/v1/classone \
  -H "Content-Type: application/json" \
  -d '{
    "model": "class-one-gemma-4-e2b",
    "state": {
      "message": "I was double billed for my subscription."
    },
    "questions": {
      "refund": {
        "type": "noul",
        "instructions": "Is customer asking for a refund?"
      },
      "routing": {
        "type": "choice",
        "instructions": "Which department?",
        "criteria": {
          "billing": "Charges and payouts",
          "technical": "App bugs and crashes"
        }
      },
      "urgency": {
        "type": "score",
        "instructions": "Ticket urgency",
        "criteria": ["low", "normal", "critical"]
      }
    }
  }'
```

### 5. Python SDK Usage

```python
from classone import ClassOneClient, Choice, Noul, Score

with ClassOneClient(base_url="http://localhost:8000") as client:
    response = client.decide(
        state="Customer account was locked after three incorrect attempts.",
        questions={
            "urgent": Noul(instructions="Is this an urgent security event?"),
            "action": Choice(
                instructions="Action required:",
                criteria={"unlock": "Send unlock link", "escalate": "Escalate to SecOps"}
            ),
            "severity": Score(
                instructions="Assess severity:",
                criteria=["low", "medium", "critical"]
            ),
        }
    )

    print("Urgent P(true):", response.nouls["urgent"].noul)
    print("Action Choice:", response.choices["action"].choice)
    print("Severity Score:", response.scores["severity"].score)
```

---

## CLI Utilities

```bash
# Run single-pass decision inference on Gemma:
python scripts/run_classone.py --model google/gemma-2-2b-it --device auto

# Benchmark latency vs traditional autoregressive LLMs:
python scripts/benchmark_latency.py --iterations 30
```

---

## Reference
- **Architecture:** [`ARCHITECTURE.md`](ARCHITECTURE.md)

---

## Legal & Attribution Notices

- Gemma is a trademark of Google LLC.
- Gemma is provided under and subject to the Gemma Terms of Use found at [ai.google.dev/gemma/terms](https://ai.google.dev/gemma/terms).
