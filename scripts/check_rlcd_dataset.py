import os

from huggingface_hub import HfApi

token = None
if os.path.exists(".env"):
    with open(".env", encoding="utf-8") as f:
        for line in f:
            if line.startswith("HF_TOKEN="):
                token = line.split("=", 1)[1].strip().strip("'\"")
                break

api = HfApi(token=token)
files = api.list_repo_files("sumleo/RLCDAlignBench", repo_type="dataset")
bench_dirs = {}
for f in files:
    if f.startswith("data/benchmarks/"):
        parts = f.split("/")
        if len(parts) >= 4:
            cat = parts[2]
            bench_dirs.setdefault(cat, []).append(parts[3])

print("10 Alignment Failure Modes & Benchmark Files:")
for cat, flist in sorted(bench_dirs.items()):
    print(f"  [{cat}] ({len(flist)} benchmarks)")
    for b in flist[:3]:
        print(f"     - {b}")
    if len(flist) > 3:
        print(f"     ... ({len(flist) - 3} more)")
