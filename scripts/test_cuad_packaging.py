import datasets

from classone.data.dataset import bundle_multi_task_items
from classone.schemas import ChoiceQuestion, NoulQuestion

print("1. Testing CUAD QA packaging...")
ds = datasets.load_dataset("chenghao/cuad_qa", split="train", streaming=True)
items = []
for r in ds:
    ctx = r.get("context", "").strip()
    q = r.get("question", "").strip()
    ans = r.get("answers", {}).get("text", [])
    has_clause = len(ans) > 0 and len(ans[0].strip()) > 0
    if not ctx or not q:
        continue

    state = {"commercial_contract": ctx[:1500], "inquired_covenant": q}
    criteria = {
        "covenant_present": "The contract explicitly contains this restrictive covenant or legal obligation.",
        "covenant_absent": "The contract does not contain or is silent regarding this covenant.",
    }
    target_choice = "covenant_present" if has_clause else "covenant_absent"

    item = bundle_multi_task_items(
        state=state,
        questions={
            "covenant_detection": ChoiceQuestion(
                instructions=f"Examine the commercial agreement and determine if the clause '{q}' is present:",
                criteria=criteria,
            ),
            "is_clause_active": NoulQuestion(instructions=f"Does the agreement contain an operative '{q}' covenant?"),
        },
        targets={
            "covenant_detection": target_choice,
            "is_clause_active": 1.0 if has_clause else 0.0,
        },
    )
    items.append(item)
    if len(items) >= 3:
        break

print(f"   [✓] Successfully formatted {len(items)} CUAD items! Sample targets: {items[0].targets}")
