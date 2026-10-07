#!/usr/bin/env python3
"""Audits missed tasks in best_souped_68pct and tests precision thresholding and dual-tier routing."""

import gc
import json
import os

import torch
from transformers import AutoTokenizer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from classone.modeling.modeling_classone import ClassOneModel
from classone.schemas import ChoiceQuestion, NoulQuestion, ScoreQuestion
from classone.tokenizer import ClassOnePromptBuilder

DATA_DIR = "data/jevbench"


def load_tier(tier_filename: str) -> list[dict]:
    path = os.path.join(DATA_DIR, tier_filename)
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate_model_on_tasks(model, builder, tasks, noul_thresh=0.5):
    results = []
    for idx, task in enumerate(tasks):
        tid = task.get("id", str(idx))
        state = task["state"]
        q_data = task["question"]
        expected = task["expected"]
        q_type = q_data.get("type", "choice")
        instructions = q_data.get("instructions", "")
        if isinstance(instructions, dict):
            instructions = json.dumps(instructions)

        if q_type == "noul":
            co_q = NoulQuestion(instructions=instructions)
        elif q_type == "choice":
            raw_crit = q_data.get("criteria", {})
            if isinstance(raw_crit, list):
                criteria = {str(i): str(c) for i, c in enumerate(raw_crit)}
            elif isinstance(raw_crit, dict):
                criteria = {
                    str(k): (v.get("what", str(v)) if isinstance(v, dict) else str(v)) for k, v in raw_crit.items()
                }
            else:
                criteria = {"yes": "yes", "no": "no"}
            co_q = ChoiceQuestion(instructions=instructions, criteria=criteria)
        elif q_type == "score":
            raw_crit = q_data.get("criteria", [])
            if isinstance(raw_crit, dict):
                criteria = [str(v) for v in raw_crit.values()]
            elif isinstance(raw_crit, list):
                criteria = [str(c) for c in raw_crit]
            else:
                criteria = ["low", "high"]
            co_q = ScoreQuestion(instructions=instructions, criteria=criteria)
        else:
            continue

        try:
            packed = builder.pack(state, {"q": co_q})
        except Exception:
            str_state = str(state)[:6000]
            packed = builder.pack(str_state, {"q": co_q})

        with torch.no_grad():
            res = model.evaluate_packed(packed)["q"]

        pred_val = None
        is_correct = False

        if q_type == "noul":
            pred_prob = float(res.noul)
            pred_val = pred_prob
            pred_bool = pred_prob >= noul_thresh
            expected_bool = str(expected).lower() in ("yes", "true", "1")
            is_correct = pred_bool == expected_bool

        elif q_type == "choice":
            pred_val = str(res.choice)
            is_correct = str(res.choice) == str(expected)

        elif q_type == "score":
            pred_val = float(res.score)
            pred_idx = int(round(res.score)) - 1
            expected_idx = int(expected)
            is_correct = pred_idx == expected_idx

        results.append(
            {
                "id": tid,
                "type": q_type,
                "expected": expected,
                "pred": pred_val,
                "correct": is_correct,
                "res": res,
            }
        )
    return results


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    easy_tasks = load_tier("easy.jsonl")
    original_tasks = load_tier("original.jsonl")
    hard_tasks = load_tier("hard.jsonl")

    print(f"Loaded: Easy ({len(easy_tasks)}), Original ({len(original_tasks)}), Hard ({len(hard_tasks)})")

    # 1. Evaluate best_souped_68pct
    model_path = "checkpoints/classone_70pct_milestone_merged"
    heads_path = "checkpoints/snapshots/best_souped_68pct/classone_heads.pt"
    print(f"\n[*] Loading best_souped_68pct from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    builder = ClassOnePromptBuilder(tokenizer)
    model = ClassOneModel.from_backbone(model_path, tokenizer=tokenizer, device=device, torch_dtype=torch.float16)
    heads = torch.load(heads_path, map_location=device)
    model.noul_head.load_state_dict(heads["noul_head"], strict=False)
    model.choice_head.load_state_dict(heads["choice_head"], strict=False)
    model.score_head.load_state_dict(heads["score_head"], strict=False)
    model.eval()

    easy_res = evaluate_model_on_tasks(model, builder, easy_tasks)
    orig_res = evaluate_model_on_tasks(model, builder, original_tasks)
    hard_res = evaluate_model_on_tasks(model, builder, hard_tasks)

    e_corr = sum(1 for r in easy_res if r["correct"])
    o_corr = sum(1 for r in orig_res if r["correct"])
    h_corr = sum(1 for r in hard_res if r["correct"])
    tot = e_corr + o_corr + h_corr

    print("best_souped_68pct baseline (thresh=0.5):")
    print(f"  Easy    : {e_corr}/{len(easy_tasks)} ({e_corr / len(easy_tasks) * 100:.1f}%)")
    print(f"  Original: {o_corr}/{len(original_tasks)} ({o_corr / len(original_tasks) * 100:.1f}%)")
    print(f"  Hard    : {h_corr}/{len(hard_tasks)} ({h_corr / len(hard_tasks) * 100:.1f}%)")
    print(f"  TOTAL   : {tot}/231 ({tot / 231 * 100:.1f}%)")

    # Audit missed tasks in Original
    print("\nMissed tasks in Original tier:")
    for r in orig_res:
        if not r["correct"]:
            print(f"  {r['type']:<7} {r['id']:<26} Exp: {r['expected']} | Pred: {r['pred']}")

    # Threshold sweep on Noul
    print("\n--- Sweeping Noul threshold on best_souped_68pct ---")
    for thresh in [0.45, 0.48, 0.50, 0.505, 0.51, 0.515, 0.52, 0.53, 0.55]:

        def recompute(r_list):
            c = 0
            for r in r_list:
                if r["type"] == "noul":
                    pred_bool = r["pred"] >= thresh
                    exp_bool = str(r["expected"]).lower() in ("yes", "true", "1")
                    if pred_bool == exp_bool:
                        c += 1
                elif r["correct"]:
                    c += 1
            return c

        ec = recompute(easy_res)
        oc = recompute(orig_res)
        hc = recompute(hard_res)
        print(
            f"  thresh={thresh:.3f}: Easy={ec}/{len(easy_tasks)} Orig={oc}/{len(original_tasks)} Hard={hc}/{len(hard_tasks)} | Total={ec + oc + hc}/231 ({(ec + oc + hc) / 231 * 100:.1f}%)"
        )

    # Clean up model
    del model, heads
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
