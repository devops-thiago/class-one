#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

from huggingface_hub import HfApi
from push_to_hub import generate_model_card, generate_lora_card

api = HfApi()

qwen_models = [
    ("devops-thiago/classone-qwen3.5-2b", "Qwen/Qwen3.5-2B"),
    ("devops-thiago/classone-qwen3.5-4b", "Qwen/Qwen3.5-4B"),
    ("devops-thiago/classone-qwen3.5-9b", "Qwen/Qwen3.5-9B"),
]

for repo_id, base_model in qwen_models:
    print(f"[*] Updating README and attribution for {repo_id}...")
    readme_content = generate_model_card(repo_id=repo_id, base_model=base_model, standalone=True)
    api.upload_file(
        path_or_fileobj=readme_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message="Fix attribution to Alibaba Cloud (Qwen Team) and verify Apache 2.0 license",
    )
    print(f"  [✓] Updated README.md on {repo_id}")

    lora_content = generate_lora_card(repo_id=repo_id, base_model=base_model)
    api.upload_file(
        path_or_fileobj=lora_content.encode("utf-8"),
        path_in_repo="lora_backbone/README.md",
        repo_id=repo_id,
        commit_message="Fix LoRA attribution to Alibaba Cloud (Qwen Team)",
    )
    print(f"  [✓] Updated lora_backbone/README.md on {repo_id}")

print("\n[✓] All Qwen model cards updated successfully on Hugging Face Hub!")
