#!/usr/bin/env python3
"""Export a Stage 2 ClassOne snapshot (merged LoRA blend + heads + tokenizer) as a standalone model directory."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import sys

import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

from classone.publishing import compute_sha256
from classone.stage2_readout import _resolve_component_adapter_dir
from classone.tokenizer import ClassOnePromptBuilder


def _apply_and_merge_backbone(backbone: torch.nn.Module, adapter_dir: pathlib.Path, scale: float) -> torch.nn.Module:
    peft_model = PeftModel.from_pretrained(backbone, str(adapter_dir))
    if scale != 1.0:
        for mod in peft_model.modules():
            if hasattr(mod, "scaling") and isinstance(mod.scaling, dict):
                for key in list(mod.scaling.keys()):
                    mod.scaling[key] = float(mod.scaling[key]) * float(scale)
    return peft_model.merge_and_unload()


def export_standalone_stage2(
    base_model: str,
    snapshot_dir: str,
    output_dir: str,
    *,
    m2_k_threshold: int = 64,
    m2_top_k: int = 16,
    device: str = "cpu",
) -> dict:
    """Merges Stage 2 LoRA component(s) into base_model and saves a standalone BF16 artifact."""
    snap_path = pathlib.Path(snapshot_dir)
    out_path = pathlib.Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    local_only = os.path.isdir(base_model) or os.environ.get("HF_HUB_OFFLINE") == "1"
    meta_path = snap_path / "snapshot_metadata.json"
    meta: dict = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    components = meta.get("merge_recipe", {}).get("components", []) or []

    print(f"[*] Loading BF16 base backbone from {base_model} on {device}...")
    backbone = AutoModel.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device,
        trust_remote_code=True,
        local_files_only=local_only,
    )

    if components:
        for comp in components:
            weight = float(comp.get("weight", 1.0))
            if weight == 0.0:
                continue
            comp_dir = _resolve_component_adapter_dir(snap_path, str(comp["name"]), weight)
            print(f"[*] Merging LoRA component {comp['name']} (weight={weight:.2f}) from {comp_dir}...")
            backbone = _apply_and_merge_backbone(backbone, comp_dir, scale=weight)
    else:
        lora_dir = snap_path / "lora_backbone"
        if not lora_dir.exists():
            raise FileNotFoundError(f"Neither merge_recipe.components nor lora_backbone found in {snap_path}")
        print(f"[*] Merging single LoRA adapter from {lora_dir}...")
        backbone = _apply_and_merge_backbone(backbone, lora_dir, scale=1.0)

    print(f"[*] Saving merged safetensors shards to {out_path}...")
    backbone.save_pretrained(str(out_path), safe_serialization=True, max_shard_size="5GB")

    heads_src = snap_path / "classone_heads.pt"
    if not heads_src.exists():
        raise FileNotFoundError(f"classone_heads.pt not found in {snap_path}")
    heads_dst = out_path / "classone_heads.pt"
    shutil.copyfile(heads_src, heads_dst)
    src_sha = compute_sha256(heads_src)
    dst_sha = compute_sha256(heads_dst)
    if src_sha != dst_sha:
        raise RuntimeError(f"SHA256 mismatch copying classone_heads.pt: {src_sha} != {dst_sha}")
    expected_sha = str(meta.get("heads_sha256", "")).strip().upper()
    if expected_sha and expected_sha != dst_sha:
        raise RuntimeError(f"Heads SHA256 mismatch vs snapshot_metadata.json: {dst_sha} != {expected_sha}")

    if meta_path.exists():
        shutil.copyfile(meta_path, out_path / "snapshot_metadata.json")

    config_file = out_path / "config.json"
    cfg_data = json.loads(config_file.read_text(encoding="utf-8"))
    cfg_data["classone_config"] = {
        "eval_mode": "stage2",
        "m2_k_threshold": int(m2_k_threshold),
        "m2_top_k": int(m2_top_k),
        "candidate": meta.get("candidate", snap_path.name),
        "heads_sha256": dst_sha,
    }
    config_file.write_text(json.dumps(cfg_data, indent=2), encoding="utf-8")

    print(f"[*] Serializing ClassOne delimiter tokens into tokenizer in {out_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        local_files_only=local_only,
    )
    ClassOnePromptBuilder(tokenizer)
    tokenizer.save_pretrained(str(out_path))

    summary = {
        "output_dir": str(out_path),
        "heads_sha256": dst_sha,
        "classone_config": cfg_data["classone_config"],
        "safetensors_files": sorted(p.name for p in out_path.glob("*.safetensors")),
    }
    print("[OK] Standalone Stage 2 artifact exported:", json.dumps(summary, indent=2))
    return summary


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Export Stage 2 merged LoRA + heads as standalone model")
    parser.add_argument("--base-model", required=True, help="Local path or HF model ID of base backbone")
    parser.add_argument("--snapshot-dir", required=True, help="Stage 2 snapshot directory")
    parser.add_argument("--output-dir", required=True, help="Destination standalone directory")
    parser.add_argument("--m2-k-threshold", type=int, default=64, help="Candidate count threshold for M2 shortlist")
    parser.add_argument("--m2-top-k", type=int, default=16, help="M2 second-pass shortlist size")
    parser.add_argument("--device", default="cpu", help="Device used while merging adapters (default: cpu)")
    args = parser.parse_args()

    export_standalone_stage2(
        base_model=args.base_model,
        snapshot_dir=args.snapshot_dir,
        output_dir=args.output_dir,
        m2_k_threshold=args.m2_k_threshold,
        m2_top_k=args.m2_top_k,
        device=args.device,
    )


if __name__ == "__main__":
    main()
