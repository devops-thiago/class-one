import json
import urllib.request

for name in ["original.jsonl", "hard.jsonl"]:
    url = f"https://raw.githubusercontent.com/fstandhartinger/jevbench/main/datasets/public/{name}"
    with urllib.request.urlopen(url) as resp:
        lines = [json.loads(line) for line in resp.read().decode("utf-8").strip().splitlines() if line.strip()]

    print(f"\n=== {name} expected formats ===")
    for qtype in ["noul", "choice", "score"]:
        sample = next((item for item in lines if item.get("question", {}).get("type") == qtype), None)
        if sample:
            exp = sample.get("expected")
            print(f"  {qtype}: expected={exp!r} (type {type(exp).__name__})")
            if qtype == "noul":
                # check variety of noul expected values
                noul_vals = {
                    str(item.get("expected")) for item in lines if item.get("question", {}).get("type") == "noul"
                }
                print(f"    all noul expected values: {noul_vals}")
            elif qtype == "score":
                score_vals = {
                    str(item.get("expected")) for item in lines if item.get("question", {}).get("type") == "score"
                }
                print(f"    all score expected values: {score_vals}")
