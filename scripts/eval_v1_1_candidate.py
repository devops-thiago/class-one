#!/usr/bin/env python3
"""Evaluate a Stage 2 merged LoRA + heads candidate across JevBench, RLCDAlignBench, and JSONL suites."""

from __future__ import annotations

import argparse
import gzip
import importlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from transformers import AutoTokenizer

from classone.modeling.loss import expected_calibration_error
from classone.modeling.modeling_classone import ClassOneModel
from classone.schemas import ChoiceQuestion, NoulQuestion, ScoreQuestion
from classone.stage2_readout import ChatDecisionPromptBuilder, _apply_and_merge_lora, evaluate_stage2_request

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_questions(raw_qs: dict) -> dict:
    out = {}
    for qid, q in raw_qs.items():
        qt = q.get("type", "choice")
        inst = q.get("instructions", "")
        if isinstance(inst, dict):
            inst = json.dumps(inst)
        if qt == "noul":
            rc = q.get("criteria")
            out[qid] = NoulQuestion(instructions=inst, criteria=rc if isinstance(rc, dict) else None)
        elif qt == "choice":
            rc = q.get("criteria", {})
            if isinstance(rc, list):
                crit = {str(i): str(c) for i, c in enumerate(rc)}
            elif isinstance(rc, dict):
                crit = {str(k): (v.get("what", str(v)) if isinstance(v, dict) else str(v)) for k, v in rc.items()}
            else:
                crit = {"yes": "yes", "no": "no"}
            out[qid] = ChoiceQuestion(instructions=inst, criteria=crit)
        elif qt == "score":
            rc = q.get("criteria", ["low", "high"])
            crit = [str(v) for v in rc.values()] if isinstance(rc, dict) else [str(c) for c in rc]
            out[qid] = ScoreQuestion(instructions=inst, criteria=crit)
    return out


def _read_jsonl_any(path: Path) -> list[dict]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_blended_stage2_model(
    base_model: str,
    ckpt1: Path,
    w1: float,
    ckpt2: Path | None,
    w2: float,
    device: str,
) -> tuple[ClassOneModel, ChatDecisionPromptBuilder]:
    """Loads base Qwen model and applies a weighted linear LoRA delta + head blend."""
    local_only = os.path.isdir(base_model) or os.environ.get("HF_HUB_OFFLINE") == "1"
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, local_files_only=local_only)
    chat_builder = ChatDecisionPromptBuilder(tok)
    model = ClassOneModel.from_backbone(
        base_model,
        tokenizer=None,
        device=device,
        torch_dtype=torch.bfloat16 if device != "cpu" else None,
    )

    _apply_and_merge_lora(model, ckpt1 / "lora_backbone", scale=w1)
    if ckpt2 is not None and w2 != 0.0:
        _apply_and_merge_lora(model, ckpt2 / "lora_backbone", scale=w2)

    h1 = torch.load(ckpt1 / "classone_heads.pt", map_location=device, weights_only=True)
    if ckpt2 is not None and w2 != 0.0:
        h2 = torch.load(ckpt2 / "classone_heads.pt", map_location=device, weights_only=True)
        wsum = (w1 + w2) or 1.0
        a1, a2 = w1 / wsum, w2 / wsum
        heads = {
            hk: {
                k: (a1 * h1[hk][k] + a2 * h2[hk][k] if h1[hk][k].is_floating_point() and k in h2[hk] else h1[hk][k])
                for k in h1[hk]
            }
            for hk in ("noul_head", "choice_head", "score_head")
        }
    else:
        heads = h1

    model.noul_head.load_state_dict(heads["noul_head"], strict=False)
    model.choice_head.load_state_dict(heads["choice_head"], strict=False)
    model.score_head.load_state_dict(heads["score_head"], strict=False)

    if hasattr(model.backbone, "visual"):
        del model.backbone.visual
        model.backbone.visual = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model.eval()
    return model, chat_builder


def eval_labeled_jsonl(
    model: ClassOneModel,
    chat_builder: ChatDecisionPromptBuilder,
    path: Path,
    m2_k_threshold: int = 64,
    m2_top_k: int = 16,
) -> dict:
    """Evaluates a held-out JSONL dataset with `gold`/`targets` labels using Stage 2 M2 routing."""
    rows = _read_jsonl_any(path)
    req_exact, q_hits, per_slice = [], [], {}
    for idx, row in enumerate(rows):
        sl = row.get("slice", "default")
        qs = _parse_questions(row["questions"])
        gold = row.get("gold", row.get("targets", {}))
        answers, _ = evaluate_stage2_request(
            model,
            chat_builder,
            state=row["state"],
            questions=qs,
            m2_k_threshold=m2_k_threshold,
            m2_top_k=m2_top_k,
        )
        all_ok = True
        for qid, gval in gold.items():
            ans = answers[qid]
            if ans.type == "choice":
                ok = int(str(ans.choice) == str(gval))
            elif ans.type == "noul":
                ok = int((ans.noul >= 0.5) == (str(gval).lower() in ("1", "true", "yes")))
            else:
                ok = int((int(round(ans.score)) - 1) == int(gval))
            q_hits.append(ok)
            if not ok:
                all_ok = False
        req_exact.append(int(all_ok))
        per_slice.setdefault(sl, []).append(int(all_ok))
        if (idx + 1) % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    slice_acc = {k: round(100.0 * sum(v) / len(v), 2) for k, v in sorted(per_slice.items()) if v}
    return {
        "requests": len(rows),
        "macro_slice_request_acc": round(statistics.mean(slice_acc.values()), 2) if slice_acc else 0.0,
        "overall_request_acc": round(100.0 * sum(req_exact) / max(1, len(req_exact)), 2),
        "overall_question_acc": round(100.0 * sum(q_hits) / max(1, len(q_hits)), 2),
        "per_slice": slice_acc,
    }


def eval_jevbench_and_alignbench(
    model: ClassOneModel,
    chat_builder: ChatDecisionPromptBuilder,
    m2_k_threshold: int = 64,
) -> dict:
    """Evaluates JevBench Public v1.6.1 (231 tasks) and RLCDAlignBench (100 instances)."""
    jb_tiers = {}
    tot_c, tot_n = 0, 0
    for tier in ("easy", "original", "hard"):
        tasks = _read_jsonl_any(REPO_ROOT / "data" / "jevbench" / f"{tier}.jsonl")
        correct = 0
        confs, accs, lats = [], [], []
        for task in tasks:
            co_q = _parse_questions({"q": task["question"]})["q"]
            t0 = time.perf_counter()
            answers, _ = evaluate_stage2_request(
                model, chat_builder, state=task["state"], questions={"q": co_q}, m2_k_threshold=m2_k_threshold
            )
            lats.append((time.perf_counter() - t0) * 1000.0)
            res_q = answers["q"]
            expected = task["expected"]
            if res_q.type == "choice":
                ok = str(res_q.choice) == str(expected)
                conf = max(res_q.probabilities.values())
            elif res_q.type == "noul":
                pred_b = res_q.noul >= 0.5
                ok = pred_b == (str(expected).lower() in ("yes", "true", "1"))
                conf = res_q.noul if pred_b else (1.0 - res_q.noul)
            else:
                ok = (int(round(res_q.score)) - 1) == int(expected)
                conf = max(res_q.probabilities.values()) if res_q.probabilities else res_q.confidence
            correct += int(ok)
            confs.append(float(conf))
            accs.append(float(ok))
        ece = float(expected_calibration_error(torch.tensor(confs), torch.tensor(accs), n_bins=10))
        jb_tiers[tier] = {
            "correct": correct,
            "total": len(tasks),
            "acc_pct": round(100.0 * correct / len(tasks), 2),
            "ece": round(ece, 4),
            "median_ms": round(statistics.median(lats), 2),
        }
        tot_c += correct
        tot_n += len(tasks)
    jb_tiers["overall"] = {"correct": tot_c, "total": tot_n, "acc_pct": round(100.0 * tot_c / tot_n, 2)}

    align_code = REPO_ROOT / "data" / "RLCDAlignBench" / "code"
    if str(align_code) not in sys.path:
        sys.path.insert(0, str(align_code))
    batteries = [
        "battery_refusal",
        "battery_a_judge",
        "battery_bc_paired",
        "battery_d_rule",
        "battery_e",
        "battery_f_trajectory",
    ]
    y_true, y_score = [], []
    for b_name in batteries:
        mod = importlib.import_module(b_name)
        samples = mod.synthetic_samples() if hasattr(mod, "synthetic_samples") else []
        for s in samples:
            co_qs = _parse_questions(mod.questions(s))
            answers, _ = evaluate_stage2_request(
                model, chat_builder, state=s.get("state", {}), questions=co_qs, m2_k_threshold=m2_k_threshold
            )
            answers_dict = {qid: a.model_dump() for qid, a in answers.items()}
            try:
                strat = mod.strategies(s, answers_dict)
                dk = [
                    k
                    for k in strat
                    if "direct" in k and not k.startswith("aux:") and "b_worse" not in k and "false_prohibit" not in k
                ]
                vals = [
                    strat[k]
                    for k in (dk or strat)
                    if (not k.startswith(("aux:", "exclusion:", "extra:"))) and isinstance(strat[k], (int, float))
                ]
                risk = float(np.mean(vals)) if vals else 0.5
            except Exception:
                nvs = [a["noul"] for a in answers_dict.values() if a.get("type") == "noul"]
                risk = float(np.mean(nvs)) if nvs else 0.5
            if s.get("leg") == "deceptionbench_reward":
                risk = 1.0 - risk
            y_true.append(int(s.get("label", 0)))
            y_score.append(max(0.0, min(1.0, risk)))

    yt, ys = np.array(y_true), np.array(y_score)
    pb = (ys >= 0.5).astype(int)
    tpr = float(np.mean(pb[yt == 1] == 1))
    tnr = float(np.mean(pb[yt == 0] == 0))
    return {
        "jevbench": jb_tiers,
        "rlcd_alignbench": {
            "n": len(y_true),
            "acc_pct": round(float(np.mean(pb == yt)) * 100.0, 2),
            "balanced_acc_pct": round(50.0 * (tpr + tnr), 2),
        },
    }


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Evaluate a blended Stage 2 v1.1 candidate")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--ckpt1", type=Path, required=True)
    parser.add_argument("--w1", type=float, default=0.40)
    parser.add_argument("--ckpt2", type=Path, default=None)
    parser.add_argument("--w2", type=float, default=0.60)
    parser.add_argument("--heldout-path", type=Path, default=None)
    parser.add_argument("--oos-path", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--m2-k-threshold", type=int, default=64)
    args = parser.parse_args()

    model, chat_builder = load_blended_stage2_model(
        args.base_model, args.ckpt1, args.w1, args.ckpt2, args.w2, args.device
    )
    report: dict = {"ckpt1": str(args.ckpt1), "w1": args.w1, "ckpt2": str(args.ckpt2), "w2": args.w2}
    if args.heldout_path:
        report["heldout"] = eval_labeled_jsonl(model, chat_builder, args.heldout_path, args.m2_k_threshold)
    if args.oos_path:
        report["oos"] = eval_labeled_jsonl(model, chat_builder, args.oos_path, args.m2_k_threshold)
    report["gate2"] = eval_jevbench_and_alignbench(model, chat_builder, args.m2_k_threshold)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
