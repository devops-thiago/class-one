import datasets

from classone.data.dataset import bundle_multi_task_items
from classone.schemas import ChoiceQuestion, NoulQuestion

print("1. Testing ContractNLI item packaging...")
ds = datasets.load_dataset("presencesw/contractnli", split="train", streaming=True)
items = []
for r in ds:
    clause = r.get("sentence1", "").strip()
    cond = r.get("sentence2", "").strip()
    lbl = str(r.get("gold_label", "neutral")).lower()
    if not clause or not cond or lbl not in ["entailment", "contradiction", "neutral"]:
        continue

    state = {"contract_clause": clause[:1200], "operational_condition": cond}
    criteria = {
        "entailment": "The operational condition is strictly required or affirmed by the contract clause.",
        "contradiction": "The operational condition directly violates or contradicts the contract clause.",
        "neutral": "The contract clause does not restrict or mandate this condition.",
    }
    target_noul = 1.0 if lbl == "entailment" else (0.0 if lbl == "contradiction" else 0.5)

    item = bundle_multi_task_items(
        state=state,
        questions={
            "clause_adjudication": ChoiceQuestion(
                instructions="Determine the legal compliance of the operational condition under the contract clause:",
                criteria=criteria,
            ),
            "is_contract_compliant": NoulQuestion(
                instructions="Is this condition compliant with and affirmed by the contract?"
            ),
        },
        targets={
            "clause_adjudication": lbl,
            "is_contract_compliant": target_noul,
        },
    )
    items.append(item)
    if len(items) >= 3:
        break

print(f"   [✓] Successfully formatted {len(items)} ContractNLI items! Sample targets: {items[0].targets}")
