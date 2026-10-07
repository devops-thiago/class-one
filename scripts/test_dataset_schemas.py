import datasets

print("Testing SNLI...")
snli = datasets.load_dataset("stanfordnlp/snli", split="train", streaming=True)
for item in snli:
    if item["label"] != -1:  # -1 is unlabeled/gold consensus missing
        print("SNLI sample:", item)
        break

print("\nTesting Banking77...")
try:
    banking = datasets.load_dataset("mteb/banking77", split="train", streaming=True)
except Exception:
    banking = datasets.load_dataset("PolyAI/banking77", split="train", streaming=True)
for item in banking:
    print("Banking77 sample:", item)
    break

print("\nTesting BeaverTails...")
bt = datasets.load_dataset("PKU-Alignment/BeaverTails", split="30k_train", streaming=True)
for item in bt:
    print("BeaverTails sample:", item)
    break
