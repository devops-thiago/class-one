#!/usr/bin/env python3
"""Creates and synchronizes the official ClassOne Hugging Face Hub Collection.

Groups all ClassOne System 1 decision models (Gemma 4 & Qwen 3.5) and curriculum
datasets under a unified public Hugging Face collection.

Usage:
    python scripts/create_hf_collection.py
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi


def load_hf_token() -> str | None:
    token = (
        os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    )
    if token:
        return token

    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN") and v:
                    return v
    return None


COLLECTION_TITLE = "ClassOne System 1 Decision Models"
COLLECTION_DESCRIPTION = (
    "ClassOne System 1 decision models (Gemma 4 & Qwen 3.5) with bilinear pointer heads "
    "calibrated via RLCD for sub-50ms high-precision decisions."
)

ITEMS = [
    {
        "item_id": "devops-thiago/classone-gemma4-e2b",
        "item_type": "model",
        "note": "ClassOne Gemma 4 E2B System 1 Decision Model (2.0B params, 128k context, 70.1% JevBench)",
    },
    {
        "item_id": "devops-thiago/classone-gemma4-e4b",
        "item_type": "model",
        "note": "ClassOne Gemma 4 E4B System 1 Decision Model (4.3B effective params, PLE architecture, 73.6% JevBench)",
    },
    {
        "item_id": "devops-thiago/classone-qwen3.5-2b",
        "item_type": "model",
        "note": "ClassOne Qwen 3.5 2B System 1 Decision Model (40.3 ms median latency, 68.0% JevBench)",
    },
    {
        "item_id": "devops-thiago/classone-qwen3.5-4b",
        "item_type": "model",
        "note": "ClassOne Qwen 3.5 4B System 1 Decision Model (70.1% JevBench, 0.046 ECE Original)",
    },
    {
        "item_id": "devops-thiago/classone-qwen3.5-9b",
        "item_type": "model",
        "note": "ClassOne Qwen 3.5 9B System 1 Decision Model (80.1% JevBench Champion Milestone)",
    },
    {
        "item_id": "devops-thiago/classone-system-one-curriculum",
        "item_type": "dataset",
        "note": "ClassOne 23,500-sample balanced alignment and decision curriculum",
    },
]


def main() -> None:
    token = load_hf_token()
    if not token:
        print("[!] Error: No Hugging Face token found in environment or .env file.")
        raise SystemExit(1)

    api = HfApi(token=token)
    user = api.whoami()["name"]
    print(f"[*] Authenticated as Hugging Face user: {user}")

    print(f"[*] Creating/retrieving collection '{COLLECTION_TITLE}' under namespace '{user}'...")
    collection = api.create_collection(
        title=COLLECTION_TITLE,
        namespace=user,
        description=COLLECTION_DESCRIPTION,
        exists_ok=True,
    )
    print(f"[✓] Collection URL: {collection.url}")
    print(f"[*] Collection Slug: {collection.slug}")

    # Inspect existing item IDs in collection to avoid duplicates
    existing_items = {item.item_id for item in collection.items}

    for item in ITEMS:
        item_id = item["item_id"]
        item_type = item["item_type"]
        note = item["note"]

        if item_id in existing_items:
            print(f"  [=] Already in collection: {item_id}")
            continue

        try:
            print(f"  [+] Adding {item_type}: {item_id}...")
            api.add_collection_item(
                collection_slug=collection.slug,
                item_id=item_id,
                item_type=item_type,
                note=note,
                exists_ok=True,
            )
            print(f"  [✓] Added {item_id}")
        except Exception as e:
            print(f"  [!] Failed to add {item_id}: {e}")

    print("\n" + "=" * 70)
    print(f"[✓] Successfully configured collection: {collection.url}")
    print("=" * 70)


if __name__ == "__main__":
    main()
