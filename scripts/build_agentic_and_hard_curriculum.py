#!/usr/bin/env python3
"""Builds the unified Real-World Agentic Tool-Routing, Security & Hard-Tier Curriculum.

Strictly uses official TRAIN splits only (zero test-set contamination):
1. NVIDIA When2Call (`nvidia/When2Call`, config=`train_pref`, split=`train`):
   - 1,200 4-way ChoiceQuestion items across `tool_call`, `cannot_answer`,
     `request_for_info`, and `direct` responses using the exact Decision Index v0.4.1
     JSON state and instruction schema.
2. Deepset Prompt Injections (`deepset/prompt-injections`, split=`train`):
   - All 546 human-labeled training examples formatted as NoulQuestion + 4-way ChoiceQuestion.
3. Hard-Tier & Composite Replay Anchors (`data/hard_tier_curriculum.jsonl`):
   - 3,500 samples (anti-splitting PO arithmetic, insurance endorsement sublimits,
     nested Boolean exceptions, anti-sycophancy, and 1,000 composite_70pct_corpus anchors).

Total: ~5,246 samples -> data/agentic_hard_combined_corpus.jsonl
"""

from collections import defaultdict
import json
import os
import random

from datasets import load_dataset

from classone.data.dataset import ClassOneDataset, bundle_multi_task_items
from classone.schemas import ChoiceQuestion, NoulQuestion
from classone.train.trainer import TrainingItem


OUTPUT_PATH = os.path.join("data", "agentic_hard_combined_corpus.jsonl")
HARD_CURRICULUM_PATH = os.path.join("data", "hard_tier_curriculum.jsonl")

WHEN2CALL_INSTRUCTIONS = (
    "Which response should the assistant give to the user's question, "
    "given the available tools?"
)


def classify_when2call_response(text: str) -> str:
    """Categorize a When2Call response string into one of the 4 canonical action classes."""
    t = text.strip()
    if "<TOOLCALL>" in t:
        return "tool_call"
    lower = t.lower()
    cannot_markers = [
        "cannot",
        "can't",
        "unable to",
        "don't have access",
        "do not have access",
        "no tool",
        "not able to",
        "apologize, but i",
        "sorry, but i",
        "beyond my capabilities",
        "don't have the ability",
    ]
    clarify_markers = [
        "could you please",
        "can you provide",
        "please specify",
        "please provide",
        "what is the",
        "which ",
        "need more information",
        "clarify",
        "to help you with",
    ]
    if any(m in lower for m in cannot_markers):
        return "cannot_answer"
    if "?" in t and any(m in lower for m in clarify_markers):
        return "request_for_info"
    return "direct"


def build_when2call_train_items(target_count: int = 1200, seed: int = 42) -> list[TrainingItem]:
    """Extracts 4-option ChoiceQuestion items strictly from `nvidia/When2Call` train_pref split."""
    rng = random.Random(seed)
    print("Loading nvidia/When2Call (config='train_pref', split='train')...")
    ds = load_dataset("nvidia/When2Call", "train_pref", split="train")

    grouped: dict[tuple[str, str], dict] = {}
    fallback_by_cat: dict[str, list[str]] = defaultdict(list)

    for row in ds:
        tools_raw = row.get("tools", [])
        messages = row.get("messages", [])
        if not messages:
            continue
        user_msg = ""
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "user":
                user_msg = str(m.get("content", "")).strip()
                break
        if not user_msg:
            continue

        chosen = row.get("chosen_response", {})
        rejected = row.get("rejected_response", {})
        chosen_text = chosen.get("content", "").strip() if isinstance(chosen, dict) else str(chosen).strip()
        rejected_text = rejected.get("content", "").strip() if isinstance(rejected, dict) else str(rejected).strip()
        if not chosen_text or not rejected_text:
            continue

        parsed_tools = []
        for t in tools_raw:
            if isinstance(t, str):
                try:
                    parsed_tools.append(json.loads(t))
                except json.JSONDecodeError:
                    parsed_tools.append(t)
            else:
                parsed_tools.append(t)

        state_str = json.dumps({"tools": parsed_tools, "question": user_msg}, ensure_ascii=False)
        key = (state_str, chosen_text)
        if key not in grouped:
            grouped[key] = {
                "state": state_str,
                "chosen": chosen_text,
                "category": classify_when2call_response(chosen_text),
                "rejected": [],
            }
        if rejected_text != chosen_text and rejected_text not in grouped[key]["rejected"]:
            grouped[key]["rejected"].append(rejected_text)

        cat_r = classify_when2call_response(rejected_text)
        if len(fallback_by_cat[cat_r]) < 500:
            fallback_by_cat[cat_r].append(rejected_text)

    by_category: dict[str, list[dict]] = defaultdict(list)
    for entry in grouped.values():
        by_category[entry["category"]].append(entry)

    for cat, entries in by_category.items():
        rng.shuffle(entries)
        print(f"  When2Call train_pref unique prompts for '{cat}': {len(entries)}")

    all_categories = ["tool_call", "cannot_answer", "request_for_info", "direct"]
    per_cat = target_count // len(all_categories)
    selected_entries = []
    for cat in all_categories:
        selected_entries.extend(by_category[cat][:per_cat])

    # Top up to target_count if any category had fewer than per_cat
    if len(selected_entries) < target_count:
        remaining = [
            e for cat in all_categories for e in by_category[cat][per_cat:]
        ]
        rng.shuffle(remaining)
        selected_entries.extend(remaining[: target_count - len(selected_entries)])

    items: list[TrainingItem] = []
    for entry in selected_entries:
        chosen_text = entry["chosen"]
        distractors = list(entry["rejected"][:3])

        # Fill up to 3 distractors from other categories if needed
        other_cats = [c for c in all_categories if c != entry["category"]]
        rng.shuffle(other_cats)
        for oc in other_cats:
            if len(distractors) >= 3:
                break
            pool = fallback_by_cat.get(oc, [])
            if pool:
                cand = rng.choice(pool)
                if cand != chosen_text and cand not in distractors:
                    distractors.append(cand)

        if len(distractors) < 3:
            continue

        options = [chosen_text] + distractors[:3]
        # Deterministic shuffle of option order to prevent position bias
        perm = list(range(4))
        rng.shuffle(perm)

        criteria = {}
        target_key = "opt_0"
        for idx, orig_idx in enumerate(perm):
            opt_key = f"opt_{idx}"
            criteria[opt_key] = options[orig_idx]
            if orig_idx == 0:
                target_key = opt_key

        cq = ChoiceQuestion(
            instructions=WHEN2CALL_INSTRUCTIONS,
            criteria=criteria,
        )
        items.append(
            bundle_multi_task_items(
                state=entry["state"],
                questions={"when2call_action": cq},
                targets={"when2call_action": target_key},
            )
        )

    print(f"Built {len(items)} 4-way When2Call training items from train_pref.")
    return items


def build_prompt_injection_train_items(seed: int = 42) -> list[TrainingItem]:
    """Extracts all 546 training examples from `deepset/prompt-injections` (split='train')."""
    rng = random.Random(seed)
    print("Loading deepset/prompt-injections (split='train')...")
    ds = load_dataset("deepset/prompt-injections", split="train")

    items: list[TrainingItem] = []
    for row in ds:
        text = str(row["text"]).strip()
        label = int(row["label"])  # 1 = injection, 0 = benign
        is_safe = 0.0 if label == 1 else 1.0

        state = f"User Input Submission:\n{text}"
        nq = NoulQuestion(
            instructions=(
                "Is this user input a safe, benign request free from prompt injection "
                "or instruction override attempts?"
            )
        )

        raw_options = [
            (
                "safe_benign",
                "Safe and benign user input: Standard question, translation, creative task, or analytical request with no instruction override attempt.",
            ),
            (
                "injection_override",
                "Adversarial prompt injection or instruction hijack: Attempts to ignore previous instructions, override system rules, hijack persona, or extract confidential system prompts.",
            ),
            (
                "policy_ambiguous",
                "Unrelated system telemetry log: Automated hardware diagnostics without user prompt content.",
            ),
            (
                "corrupted_payload",
                "Malformed binary encoding: Non-textual byte stream lacking semantic instructions.",
            ),
        ]
        target_orig = "injection_override" if label == 1 else "safe_benign"
        rng.shuffle(raw_options)
        criteria = {k: desc for k, desc in raw_options}

        cq = ChoiceQuestion(
            instructions="Classify the security posture of the user input submission.",
            criteria=criteria,
        )

        items.append(
            bundle_multi_task_items(
                state=state,
                questions={"is_safe_input": nq, "security_posture": cq},
                targets={"is_safe_input": is_safe, "security_posture": target_orig},
            )
        )

    print(f"Built {len(items)} Prompt Injection training items from train split.")
    return items


def main() -> None:
    rng = random.Random(42)
    all_items: list[TrainingItem] = []

    # 1. Load existing Hard-Tier + Composite Replay Curriculum (3,500 items)
    if os.path.exists(HARD_CURRICULUM_PATH):
        hard_ds = ClassOneDataset.load_jsonl(HARD_CURRICULUM_PATH)
        print(f"Loaded {len(hard_ds.items)} items from {HARD_CURRICULUM_PATH}")
        all_items.extend(hard_ds.items)
    else:
        raise FileNotFoundError(f"Missing required {HARD_CURRICULUM_PATH}")

    # 2. Add NVIDIA When2Call train_pref items (1,200 items)
    w2c_items = build_when2call_train_items(target_count=1200, seed=42)
    all_items.extend(w2c_items)

    # 3. Add Deepset Prompt Injections train split items (546 items)
    pi_items = build_prompt_injection_train_items(seed=42)
    all_items.extend(pi_items)

    rng.shuffle(all_items)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    dataset = ClassOneDataset(all_items)
    dataset.save_jsonl(OUTPUT_PATH)
    print(f"Saved {len(all_items)} unified training items to {OUTPUT_PATH}")

    # Also save a high-density 1,646-item Agentic + Security + Replay micro-curriculum
    micro_path = os.path.join("data", "agentic_security_replay_1600.jsonl")
    rng_micro = random.Random(42)
    hard_anchors = list(hard_ds.items)
    rng_micro.shuffle(hard_anchors)
    micro_items = list(w2c_items[:700]) + list(pi_items) + hard_anchors[:400]
    rng_micro.shuffle(micro_items)
    ClassOneDataset(micro_items).save_jsonl(micro_path)
    print(f"Saved {len(micro_items)} high-density micro-curriculum items to {micro_path}")


if __name__ == "__main__":
    main()
