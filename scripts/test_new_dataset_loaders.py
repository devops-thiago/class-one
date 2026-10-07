import json
import zipfile

import datasets
from huggingface_hub import hf_hub_download

print("1. Testing pminervini/HaluEval (qa)...")
try:
    ds_halu = datasets.load_dataset("pminervini/HaluEval", "qa", split="data", streaming=True)
    for row in ds_halu:
        print("   HaluEval sample keys:", list(row.keys()))
        print("   Question:", row["question"][:80])
        print("   Right:", row["right_answer"][:60])
        print("   Hallucinated:", row["hallucinated_answer"][:60])
        break
    print("   [✓] HaluEval OK!")
except Exception as e:
    print("   [!] HaluEval failed:", e)

print("\n2. Testing Anthropic/model-written-evals (sycophancy)...")
try:
    syc_path = hf_hub_download(
        repo_id="Anthropic/model-written-evals",
        repo_type="dataset",
        filename="sycophancy/sycophancy_on_nlp_survey.jsonl",
    )
    with open(syc_path, encoding="utf-8") as f:
        first_line = json.loads(f.readline())
        print("   Sycophancy sample keys:", list(first_line.keys()))
        print("   Question snippet:", first_line["question"][:100])
        print("   Match (sycophantic):", first_line.get("answer_matching_behavior"))
        print("   Not match (truthful):", first_line.get("answer_not_matching_behavior"))
    print("   [✓] Anthropic Sycophancy OK!")
except Exception as e:
    print("   [!] Anthropic Sycophancy failed:", e)

print("\n3. Testing tau/scrolls (contract_nli)...")
try:
    zip_path = hf_hub_download(repo_id="tau/scrolls", repo_type="dataset", filename="contract_nli.zip")
    with zipfile.ZipFile(zip_path, "r") as z:
        print("   Zip contents:", z.namelist()[:5])
        with z.open("train.jsonl") as f:
            first_row = json.loads(f.readline().decode("utf-8"))
            print("   Contract NLI keys:", list(first_row.keys()))
            print("   Input preview:", first_row["input"][:100])
            print("   Output:", first_row["output"])
    print("   [✓] tau/scrolls Contract NLI OK!")
except Exception as e:
    print("   [!] tau/scrolls failed:", e)
