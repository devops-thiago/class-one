import datasets

print("Testing 1: 3nesdeniz/agentic-prompt-injection-boundary-pairs...")
try:
    ds1 = datasets.load_dataset("3nesdeniz/agentic-prompt-injection-boundary-pairs", split="train", streaming=True)
    for row in ds1:
        print("  Sample keys:", list(row.keys()))
        print("  Sample preview:", {k: str(v)[:80] for k, v in row.items()})
        break
except Exception as e:
    print("  Failed to load ds1:", e)

print("\nTesting 2: kiddothe2b/contract-nli...")
try:
    ds2 = datasets.load_dataset("kiddothe2b/contract-nli", split="train", streaming=True)
    for row in ds2:
        print("  Sample keys:", list(row.keys()))
        print("  Sample preview:", {k: str(v)[:80] for k, v in row.items()})
        break
except Exception as e:
    print("  Failed to load ds2:", e)

print("\nTesting 3: ErikYip/LLM-Uncertainty-Bench...")
try:
    ds3 = datasets.load_dataset("ErikYip/LLM-Uncertainty-Bench", split="train", streaming=True)
    for row in ds3:
        print("  Sample keys:", list(row.keys()))
        print("  Sample preview:", {k: str(v)[:80] for k, v in row.items()})
        break
except Exception as e:
    print("  Failed to load ds3:", e)

print("\nTesting 4: allenai/sciq...")
try:
    ds4 = datasets.load_dataset("allenai/sciq", split="train", streaming=True)
    for row in ds4:
        print("  Sample keys:", list(row.keys()))
        print("  Sample preview:", {k: str(v)[:80] for k, v in row.items()})
        break
except Exception as e:
    print("  Failed to load ds4:", e)
