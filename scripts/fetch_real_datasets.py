#!/usr/bin/env python3
"""Builds a rich multi-domain training corpus using real public datasets from Hugging Face:

1. stanfordnlp/snli (1,500 pairs) -> NLI & Factuality (entailment, contradiction, neutral)
2. mteb/banking77 (1,500 queries) -> Fine-grained Intent Classification across 77 categories
3. PKU-Alignment/BeaverTails (1,500 pairs) -> Safety, Refusal, and Harmfulness Discrimination

Exports to `data/training_corpus.jsonl`.
"""

import os
import random

import datasets

from classone.data.dataset import ClassOneDataset, bundle_multi_task_items
from classone.schemas import ChoiceQuestion, NoulQuestion, ScoreQuestion

OUTPUT_PATH = "data/training_corpus.jsonl"


def fetch_snli_items(target_count: int = 1500) -> list:
    print(f"[*] Streaming {target_count} samples from stanfordnlp/snli...")
    ds = datasets.load_dataset("stanfordnlp/snli", split="train", streaming=True)
    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
    score_map = {0: 3, 1: 2, 2: 1}  # 1-3 scale
    noul_map = {0: 1.0, 1: 0.0, 2: 0.0}

    items = []
    for row in ds:
        lbl = row.get("label", -1)
        if lbl not in label_map:
            continue

        premise = row.get("premise", "").strip()
        hypothesis = row.get("hypothesis", "").strip()
        if not premise or not hypothesis:
            continue

        state = {"premise": premise, "hypothesis": hypothesis}
        item = bundle_multi_task_items(
            state=state,
            questions={
                "relation": ChoiceQuestion(
                    instructions="Determine the logical relationship between premise and hypothesis:",
                    criteria={
                        "entailment": "Hypothesis is definitely true given the premise",
                        "neutral": "Hypothesis might be true or false given the premise",
                        "contradiction": "Hypothesis contradicts the premise",
                    },
                ),
                "is_entailed": NoulQuestion(instructions="Does the premise strictly entail the hypothesis?"),
                "agreement_level": ScoreQuestion(
                    instructions="Rate logical agreement from contradiction to entailment:",
                    criteria=["contradiction", "neutral", "strict_entailment"],
                ),
            },
            targets={
                "relation": label_map[lbl],
                "is_entailed": noul_map[lbl],
                "agreement_level": score_map[lbl],
            },
        )
        items.append(item)
        if len(items) >= target_count:
            break

    print(f"    Loaded {len(items)} SNLI items.")
    return items


def fetch_banking77_items(target_count: int = 1500) -> list:
    print(f"[*] Streaming {target_count} samples from mteb/banking77...")
    try:
        ds = datasets.load_dataset("mteb/banking77", split="train", streaming=True)
    except Exception:
        ds = datasets.load_dataset("PolyAI/banking77", split="train", streaming=True)

    # First pass: collect unique labels and descriptions
    all_rows = []
    all_label_texts = set()
    for row in ds:
        lbl_text = row.get("label_text")
        if not lbl_text and "label" in row:
            lbl_text = f"intent_{row['label']}"
        all_rows.append((row["text"].strip(), lbl_text))
        all_label_texts.add(lbl_text)
        if len(all_rows) >= target_count * 2:
            break

    label_list = sorted(all_label_texts)

    items = []
    for text, true_lbl in all_rows[:target_count]:
        if not text:
            continue

        # Sample 3 negative distractors
        distractors = [lbl for lbl in label_list if lbl != true_lbl]
        sampled_distractors = random.sample(distractors, min(3, len(distractors)))
        candidates = [true_lbl] + sampled_distractors
        random.shuffle(candidates)

        criteria = {c: c.replace("_", " ").title() for c in candidates}

        # Check if urgent intent
        is_urgent = 1.0 if any(term in true_lbl for term in ["lost", "stolen", "compromise", "fraud"]) else 0.0

        state = {"customer_query": text}
        item = bundle_multi_task_items(
            state=state,
            questions={
                "intent": ChoiceQuestion(
                    instructions="Identify the customer's primary banking intent:",
                    criteria=criteria,
                ),
                "is_urgent": NoulQuestion(instructions="Is this query urgent regarding card loss, fraud, or theft?"),
            },
            targets={
                "intent": true_lbl,
                "is_urgent": is_urgent,
            },
        )
        items.append(item)

    print(f"    Loaded {len(items)} Banking77 items.")
    return items


def fetch_beavertails_items(target_count: int = 1500) -> list:
    print(f"[*] Streaming {target_count} samples from PKU-Alignment/BeaverTails...")
    ds = datasets.load_dataset("PKU-Alignment/BeaverTails", split="30k_train", streaming=True)

    items = []
    for row in ds:
        prompt = row.get("prompt", "").strip()
        response = row.get("response", "").strip()
        is_safe = bool(row.get("is_safe", True))

        if not prompt or not response:
            continue

        # Extract active violation categories if unsafe
        cats = row.get("category", {})
        active_cats = [k for k, v in cats.items() if v]
        active_cats[0].replace("_", " ") if active_cats else ("safe" if is_safe else "harmful")

        state = {"prompt": prompt, "model_response": response}
        item = bundle_multi_task_items(
            state=state,
            questions={
                "is_safe": NoulQuestion(instructions="Is the model response safe, ethical, and non-harmful?"),
                "safety_verdict": ChoiceQuestion(
                    instructions="Determine the safety classification of the response:",
                    criteria={
                        "safe": "Safe and compliant response",
                        "harmful": "Unsafe response violating safety principles",
                        "borderline": "Ambiguous response needing caution",
                    },
                ),
                "severity_score": ScoreQuestion(
                    instructions="Rate the safety risk score from benign to severe harm:",
                    criteria=["safe_benign", "mild_concern", "severe_harm"],
                ),
            },
            targets={
                "is_safe": 1.0 if is_safe else 0.0,
                "safety_verdict": "safe" if is_safe else "harmful",
                "severity_score": 1 if is_safe else 3,
            },
        )
        items.append(item)
        if len(items) >= target_count:
            break

    print(f"    Loaded {len(items)} BeaverTails items.")
    return items


def main():
    os.makedirs("data", exist_ok=True)
    all_items = []

    # 1. SNLI (NLI & Factuality)
    all_items.extend(fetch_snli_items(target_count=1500))

    # 2. Banking77 (Intent Classification)
    all_items.extend(fetch_banking77_items(target_count=1500))

    # 3. BeaverTails (Toxicity & Refusal)
    all_items.extend(fetch_beavertails_items(target_count=1500))

    random.shuffle(all_items)
    print(f"\n[*] Total compiled dataset size: {len(all_items)} real human-annotated items.")

    ClassOneDataset(all_items).save_jsonl(OUTPUT_PATH)
    print(f"[✓] Successfully exported multi-domain corpus to {OUTPUT_PATH}!")


if __name__ == "__main__":
    main()
