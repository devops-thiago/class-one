#!/usr/bin/env python3
"""Official JevBench v1.6.1 Offline GPU Runner Adapter for ClassOne System 1.

Conforms to jevbench.adapters specification for native in-process GPU execution,
eliminating the 2x web demo latency penalty and network overhead.

Supports:
- Direct local GPU inference (Qwen 3.5 9B / Gemma 4 E2B / E4B)
- Native option probabilities via Bilinear Interaction Scorer [q; k; |q-k|; q*k]
- Latent depth pause tokens (<|pause|>) for complex multi-hop constraint tracking
- Strict JevBench v1.6.1 probability normalization & ECE compliance
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

try:
    from jevbench.adapters.base import DecisionResult, NOUL_LABELS
except ImportError:
    NOUL_LABELS = ["no", "yes"]

    @dataclass
    class DecisionResult:
        adapter: str
        ok: bool
        probs: Optional[dict[str, float]] = None
        probs_source: str = "native"
        model: str = ""
        status: Optional[int] = None
        error: Optional[str] = None
        latency_s: float = 0.0
        usage: dict[str, int] = field(default_factory=dict)
        raw: Optional[Any] = None
        request_body: Optional[Any] = None
        label: Optional[str] = None

        def to_public(self) -> dict[str, Any]:
            return {
                "adapter": self.adapter,
                "ok": self.ok,
                "probs": self.probs,
                "probs_source": self.probs_source,
                "model": self.model,
                "status": self.status,
                "error": self.error,
                "latency_s": self.latency_s,
                "usage": self.usage,
                "label": self.label,
            }


class ClassOneAdapter:
    """Native offline GPU runner adapter for ClassOne System One decision models."""

    name = "classone_local"
    cost_basis = "self_hosted_gpu_estimated_input_tokens"

    def __init__(
        self,
        model_path: str | None = None,
        heads_path: str | None = None,
        device: str | None = None,
        quantization: str | None = "4bit",
        pause_tokens: int = 8,
        price_input_per_m: float = 0.042,
        price_output_per_m: float = 0.0,
        timeout_s: float = 60.0,
    ):
        # Default champion model path
        default_model = "worktrees/qwen-3.5-9b/checkpoints/classone_qwen9b_champion_merged"
        if not os.path.exists(default_model):
            default_model = "devops-thiago/classone-qwen3.5-9b"

        # Default calibrated candidate heads
        default_heads = "checkpoints/qwen9b_top3_candidate/classone_heads.pt"
        if not os.path.exists(default_heads):
            default_heads = os.path.join(default_model, "classone_heads.pt")

        self.model_path = model_path or os.environ.get("CLASSONE_MODEL", default_model)
        self.heads_path = heads_path or os.environ.get("CLASSONE_HEADS", default_heads)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.quantization = quantization
        self.pause_tokens = pause_tokens
        self.price_input_per_m = price_input_per_m
        self.price_output_per_m = price_output_per_m
        self.timeout_s = timeout_s
        self.model_name = os.path.basename(self.model_path.rstrip("/\\")) or "classone-qwen3.5-9b"

        self._loaded = None
        self.load_s = 0.0

    def prepare(self, task: Any = None) -> None:
        """Pre-warms model weights and CUDA execution kernels outside the benchmark timer.

        Conforms to JevBench runner protocol: if hasattr(adapter, 'prepare'), runner calls
        prepare(t) before starting the benchmark clock.
        """
        self.load()

    def load(self):
        """Loads ClassOne model, tokenizer, and decision heads into GPU memory."""
        if self._loaded is not None:
            return self._loaded

        started = time.perf_counter()
        from transformers import AutoTokenizer
        from classone.modeling.modeling_classone import ClassOneModel
        from classone.tokenizer import ClassOnePromptBuilder

        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        builder = ClassOnePromptBuilder(tokenizer, default_pause_tokens=self.pause_tokens)

        # Load ClassOne model with calibrated decision heads
        model = ClassOneModel.from_backbone(
            base_model_name_or_path=self.model_path,
            tokenizer=tokenizer,
            device=self.device,
            torch_dtype=torch.bfloat16 if self.device.startswith("cuda") else torch.float32,
            quantization=self.quantization if self.device.startswith("cuda") else None,
        )

        # If custom heads provided, load them with backward compatibility
        target_heads = self.heads_path
        if not target_heads or not os.path.exists(target_heads):
            target_heads = os.path.join(self.model_path, "classone_heads.pt")

        if target_heads and os.path.exists(target_heads):
            heads_data = torch.load(target_heads, map_location=self.device)
            model.noul_head.load_state_dict(heads_data.get("noul_head", {}), strict=False)
            model.choice_head.load_state_dict(heads_data.get("choice_head", {}), strict=False)
            model.score_head.load_state_dict(heads_data.get("score_head", {}), strict=False)

        model.eval()

        # Warm up CUDA kernels with a dummy forward pass to eliminate first-token JIT latency
        if self.device.startswith("cuda") and torch.cuda.is_available():
            try:
                from classone.schemas import NoulQuestion
                dummy_packed = builder.pack(
                    "System warmup state.",
                    {"warmup_q": NoulQuestion(instructions="Warmup test")},
                    num_pause_tokens=self.pause_tokens,
                )
                with torch.inference_mode():
                    model.evaluate_batch([dummy_packed], position_invariant=True)
                torch.cuda.synchronize()
            except Exception:
                pass

        self.load_s = time.perf_counter() - started
        self._loaded = (tokenizer, builder, model)
        return self._loaded

    @staticmethod
    def build_request(task: Any) -> dict[str, Any]:
        """Parses a JevBench task into normalized state, question, and criteria."""
        state = getattr(task, "state", task.get("state") if isinstance(task, dict) else "")
        if not isinstance(state, str):
            state = json.dumps(state, ensure_ascii=False)

        raw_q = getattr(task, "question", task.get("question") if isinstance(task, dict) else {})
        q_type = raw_q.get("type", "choice")
        instructions = raw_q.get("instructions", "")
        if isinstance(instructions, dict):
            instructions = json.dumps(instructions, ensure_ascii=False)

        labels = getattr(task, "labels", None)
        if labels is None and isinstance(task, dict):
            labels = task.get("labels", [])

        criteria = raw_q.get("criteria", {})
        options = {}

        if q_type == "noul":
            if not labels:
                labels = ["no", "yes"]
            if isinstance(criteria, dict):
                options = {
                    "no": str(criteria.get("false", criteria.get("no", "No / False"))),
                    "yes": str(criteria.get("true", criteria.get("yes", "Yes / True"))),
                }
            else:
                options = {"no": "No condition false", "yes": "Yes condition true"}

        elif q_type == "score":
            if isinstance(criteria, list):
                labels = [str(i + 1) for i in range(len(criteria))]
                options = {str(i + 1): str(c) for i, c in enumerate(criteria)}
            elif isinstance(criteria, dict):
                labels = [str(k) for k in criteria.keys()]
                options = {str(k): str(v) for k, v in criteria.items()}
            else:
                labels = ["1", "2", "3", "4", "5"]
                options = {k: f"Level {k}" for k in labels}

        else:  # choice
            if isinstance(criteria, dict):
                labels = list(criteria.keys())
                options = {
                    str(k): (v.get("what", str(v)) if isinstance(v, dict) else str(v))
                    for k, v in criteria.items()
                }
            elif isinstance(criteria, list):
                labels = [str(i) for i in range(len(criteria))]
                options = {str(i): str(c) for i, c in enumerate(criteria)}
            else:
                labels = ["A", "B"]
                options = {"A": "Option A", "B": "Option B"}

        return {
            "state": state,
            "q_type": q_type,
            "instructions": instructions,
            "criteria": options,
            "labels": [str(l) for l in labels],
        }

    def run(self, task: Any) -> DecisionResult:
        """Executes native feedforward decision inference on local GPU."""
        result = DecisionResult(
            adapter=self.name,
            ok=False,
            probs_source="native",
            model=self.model_name,
        )
        request = self.build_request(task)
        result.request_body = request

        # Ensure model is already loaded and pre-warmed BEFORE starting decision timer
        tokenizer, builder, model = self.load()
        from classone.schemas import ChoiceQuestion, NoulQuestion, ScoreQuestion

        q_type = request["q_type"]
        instr = request["instructions"]
        crit = request["criteria"]
        qid = "q_eval"

        if q_type == "noul":
            questions = {qid: NoulQuestion(instructions=instr)}
        elif q_type == "score":
            levels = [crit[k] for k in request["labels"]]
            questions = {qid: ScoreQuestion(instructions=instr, criteria=levels)}
        else:
            questions = {qid: ChoiceQuestion(instructions=instr, criteria=crit)}

        packed = builder.pack(
            request["state"],
            questions,
            num_pause_tokens=self.pause_tokens,
        )

        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

        started = time.perf_counter()
        try:
            with torch.inference_mode():
                answers = model.evaluate_batch([packed], position_invariant=True)

            if self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()

            result.latency_s = time.perf_counter() - started
            res = answers[0][qid]

            # Format normalized distribution
            probs_dict = {}
            if q_type == "noul":
                p_yes = float(res.noul)
                p_no = float(1.0 - p_yes)
                probs_dict = {"no": p_no, "yes": p_yes}
                best_label = "yes" if p_yes >= 0.5 else "no"
            elif q_type == "score":
                probs_dict = {str(k): float(v) for k, v in res.probabilities.items()}
                best_label = str(max(probs_dict.items(), key=lambda kv: kv[1])[0])
            else:
                probs_dict = {str(k): float(v) for k, v in res.probabilities.items()}
                best_label = str(res.choice)

            # Ensure sum to 1.0 within JevBench tolerance
            total_p = sum(probs_dict.values())
            if total_p > 0:
                probs_dict = {k: max(0.0, v / total_p) for k, v in probs_dict.items()}

            result.probs = probs_dict
            result.label = best_label
            result.usage = {
                "input_tokens": int(packed.input_ids.shape[1]),
                "output_tokens": 0,
            }
            result.raw = {
                "runtime": {
                    "checkpoint": self.model_name,
                    "generated_tokens": 0,
                    "max_input_tokens": 4096,
                    "probability_origin": "classone-bilinear-heads-native",
                    "load_s": self.load_s,
                    "pause_tokens": self.pause_tokens,
                }
            }
            result.ok = True

        except Exception as exc:
            if self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            result.latency_s = time.perf_counter() - started
            result.error = f"{type(exc).__name__}: {str(exc)[:300]}"

        return result

    def reserve_estimate(self, task: Any) -> float:
        return 0.0


# Entry point for direct adapter evaluation
if __name__ == "__main__":
    adapter = ClassOneAdapter()
    print(f"[*] Initialized {adapter.name} targeting model: {adapter.model_path}")
    test_task = {
        "state": "Customer contract signed on Jan 15, 2026. Cancellation submitted within 10 days.",
        "question": {
            "type": "noul",
            "instructions": "Is the cancellation eligible for 100% refund under 30-day policy?",
        },
        "labels": ["no", "yes"],
    }
    adapter.prepare(test_task)
    print(f"[*] Model loaded and warmed in {adapter.load_s:.2f}s")
    latencies = []
    for i in range(3):
        res = adapter.run(test_task)
        lat_ms = res.latency_s * 1000.0
        latencies.append(lat_ms)
        print(f"[✓] Run {i+1}: ok={res.ok}, latency={lat_ms:.2f}ms, probs={res.probs}")
    print(f"[★] Mean GPU Latency: {sum(latencies)/len(latencies):.2f}ms | p50: {sorted(latencies)[1]:.2f}ms")
