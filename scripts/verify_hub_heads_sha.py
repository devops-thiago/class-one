#!/usr/bin/env python3
"""Verifies that README.md and benchmarks/baselines.json match live HF Hub classone_heads.pt LFS SHA256s."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

from classone.publishing import load_baseline_registry, parse_readme_heads_hashes

REPO_ROOT = Path(__file__).resolve().parents[1]


def verify_hub_heads(
    *,
    readme_path: Path = REPO_ROOT / "README.md",
    baselines_path: Path = REPO_ROOT / "benchmarks" / "baselines.json",
    api: HfApi | None = None,
) -> list[str]:
    """Compares README.md, baselines.json, and live HF Hub LFS SHA256 values."""
    errors: list[str] = []
    readme_hashes = parse_readme_heads_hashes(readme_path.read_text(encoding="utf-8"))
    registry = load_baseline_registry(baselines_path)
    registered_models = registry.get("models", {})

    if not readme_hashes:
        return [f"No SHA256 rows found in {readme_path}"]

    for repo_id, entry in registered_models.items():
        if repo_id not in readme_hashes:
            errors.append(f"{repo_id}: missing from README.md SHA256 table")
        else:
            expected_reg = str(entry["heads_sha256"]).upper()
            if readme_hashes[repo_id] != expected_reg:
                errors.append(f"{repo_id}: README SHA256 ({readme_hashes[repo_id]}) != baselines.json ({expected_reg})")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    client = api if api is not None else HfApi(token=token)

    print(f"Checking {len(readme_hashes)} README hashes against Hugging Face Hub LFS metadata...")
    for repo_id, readme_sha in readme_hashes.items():
        paths_info = client.get_paths_info(repo_id=repo_id, paths=["classone_heads.pt"])
        if not paths_info or getattr(paths_info[0], "lfs", None) is None:
            errors.append(f"{repo_id}: classone_heads.pt LFS metadata unavailable on Hub")
            print(f"MISSING  {repo_id}")
            continue

        hub_sha = str(paths_info[0].lfs.sha256).upper()
        hub_size = getattr(paths_info[0], "size", "unknown")
        if readme_sha == hub_sha:
            print(f"OK       {repo_id}")
            print(f"  README {readme_sha}")
            print(f"  Hub    {hub_sha} ({hub_size} bytes)")
        else:
            print(f"MISMATCH {repo_id}")
            print(f"  README {readme_sha}")
            print(f"  Hub    {hub_sha} ({hub_size} bytes)")
            errors.append(f"{repo_id}: README SHA256 ({readme_sha}) != Hub LFS SHA256 ({hub_sha})")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify README/baseline SHA256 hashes match Hugging Face Hub")
    parser.add_argument("--readme", type=Path, default=REPO_ROOT / "README.md")
    parser.add_argument("--baselines", type=Path, default=REPO_ROOT / "benchmarks" / "baselines.json")
    args = parser.parse_args()

    errors = verify_hub_heads(readme_path=args.readme, baselines_path=args.baselines)
    if errors:
        print("\nFAILED:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("\nPASSED: all README, baselines.json, and Hugging Face Hub LFS SHA256 values match.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
