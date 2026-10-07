import json
import random
import re
import urllib.request

import datasets

from classone.data.dataset import bundle_multi_task_items
from classone.schemas import ChoiceQuestion, NoulQuestion

print("1. Testing DROP item formatting...")
ds_drop = datasets.load_dataset("ucinlp/drop", split="train", streaming=True)
drop_items = []
for row in ds_drop:
    spans = row.get("answers_spans", {}).get("spans", [])
    if not spans:
        continue
    true_ans = spans[0].strip()
    # Check if number
    if re.match(r"^-?\d+(\.\d+)?$", true_ans):
        val = int(float(true_ans))
        distractors = [str(val - 1), str(val + 1), str(val + 2)]
    else:
        distractors = ["Option None", "Insufficient context", "Zero"]

    candidates = [true_ans] + distractors
    random.shuffle(candidates)
    criteria = {c: f"Calculated result: {c}" for c in candidates}

    item = bundle_multi_task_items(
        state={"passage": row["passage"][:1500], "question": row["question"]},
        questions={
            "result": ChoiceQuestion(
                instructions="Calculate the answer strictly from the passage facts:", criteria=criteria
            ),
            "is_determinable": NoulQuestion(instructions="Is this question definitively answerable from the passage?"),
        },
        targets={"result": true_ans, "is_determinable": 1.0},
    )
    drop_items.append(item)
    if len(drop_items) >= 2:
        break

print(f"   [✓] Successfully formatted {len(drop_items)} DROP items!")

print("\n2. Testing are_you_sure sycophancy item formatting...")
url = "https://raw.githubusercontent.com/meg-tong/sycophancy-eval/main/datasets/are_you_sure.jsonl"
req = urllib.request.Request(url, headers={"User-Agent": "classone"})
syc_items = []
with urllib.request.urlopen(req) as resp:
    for _ in range(5):
        line = resp.readline().decode("utf-8")
        if not line:
            break
        row = json.loads(line)
        base = row.get("base", {})
        q_text = base.get("question", "")
        correct = base.get("correct_letter", "")
        answers_raw = base.get("answers", "")

        # Parse options
        opts = re.findall(r"\(([A-E])\)\s*([^\n\r]+)", answers_raw)
        if len(opts) < 2:
            continue

        criteria = {letter: text.strip() for letter, text in opts}
        if correct not in criteria:
            continue

        item = bundle_multi_task_items(
            state={"problem": q_text, "dialogue": str(row.get("prompt", ""))[:1200]},
            questions={
                "resilient_answer": ChoiceQuestion(
                    instructions="Select the objectively correct answer, maintaining factual conviction even when challenged:",
                    criteria=criteria,
                ),
                "is_factually_certain": NoulQuestion(
                    instructions="Is this solution mathematically verified and certain?"
                ),
            },
            targets={"resilient_answer": correct, "is_factually_certain": 1.0},
        )
        syc_items.append(item)
        if len(syc_items) >= 2:
            break

print(f"   [✓] Successfully formatted {len(syc_items)} Sycophancy items!")
