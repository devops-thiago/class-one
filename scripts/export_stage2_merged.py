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
from classone.stage2_readout import (
    _resolve_component_adapter_dir,
    detect_model_family,
    strip_unused_multimodal_towers,
)
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
    snapshot_dir: str | None = None,
    output_dir: str = "",
    *,
    heads_path: str | None = None,
    expected_heads_sha256: str | None = None,
    allow_no_lora: bool = False,
    strip_multimodal: bool = False,
    codebook_file: str | None = None,
    candidate_name: str | None = None,
    base_model_revision: str | None = None,
    m2_k_threshold: int = 64,
    m2_top_k: int = 16,
    device: str = "cpu",
) -> dict:
    """Merges Stage 2 LoRA component(s) (or raw base when allow_no_lora=True) and saves a standalone BF16 artifact."""
    if not output_dir:
        raise ValueError("output_dir must be a non-empty path.")
    if snapshot_dir is None and not (allow_no_lora and heads_path):
        raise ValueError("snapshot_dir is required unless allow_no_lora=True and heads_path is provided.")

    snap_path = pathlib.Path(snapshot_dir) if snapshot_dir else None
    out_path = pathlib.Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    local_only = os.path.isdir(base_model) or os.environ.get("HF_HUB_OFFLINE") == "1"
    meta_path = snap_path / "snapshot_metadata.json" if snap_path is not None else None
    meta: dict = (
        json.loads(meta_path.read_text(encoding="utf-8")) if (meta_path is not None and meta_path.exists()) else {}
    )
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
        assert snap_path is not None
        for comp in components:
            weight = float(comp.get("weight", 1.0))
            if weight == 0.0:
                continue
            comp_dir = _resolve_component_adapter_dir(snap_path, str(comp["name"]), weight)
            print(f"[*] Merging LoRA component {comp['name']} (weight={weight:.2f}) from {comp_dir}...")
            backbone = _apply_and_merge_backbone(backbone, comp_dir, scale=weight)
    elif snap_path is not None and (snap_path / "lora_backbone").exists():
        lora_dir = snap_path / "lora_backbone"
        print(f"[*] Merging single LoRA adapter from {lora_dir}...")
        backbone = _apply_and_merge_backbone(backbone, lora_dir, scale=1.0)
    elif allow_no_lora:
        print("[*] Skipping LoRA merge (allow_no_lora=True; exporting unmodified base weights)...")
    else:
        raise FileNotFoundError(f"Neither merge_recipe.components nor lora_backbone found in {snap_path}")

    if strip_multimodal:
        if getattr(backbone, "language_model", None) is not None and detect_model_family(backbone) == "gemma4":
            print("[*] Stripping Gemma 4 vision/audio towers -> exporting Gemma4TextModel (language_model)...")
            backbone = backbone.language_model
        else:
            stripped = strip_unused_multimodal_towers(backbone, clear_cuda_cache=False)
            if stripped:
                print(f"[*] Stripped unused multimodal towers: {stripped}")

    print(f"[*] Saving safetensors shards to {out_path}...")
    backbone.save_pretrained(str(out_path), safe_serialization=True, max_shard_size="5GB")

    if heads_path:
        heads_src = pathlib.Path(heads_path)
    elif snap_path is not None:
        heads_src = snap_path / "classone_heads.pt"
    else:
        raise FileNotFoundError("heads_path or snapshot_dir with classone_heads.pt is required.")
    if not heads_src.exists():
        raise FileNotFoundError(f"classone_heads.pt not found at {heads_src}")

    heads_dst = out_path / "classone_heads.pt"
    shutil.copyfile(heads_src, heads_dst)
    src_sha = compute_sha256(heads_src)
    dst_sha = compute_sha256(heads_dst)
    if src_sha != dst_sha:
        raise RuntimeError(f"SHA256 mismatch copying classone_heads.pt: {src_sha} != {dst_sha}")
    meta_sha = str(meta.get("heads_sha256", "")).strip().upper()
    if meta_sha and meta_sha != dst_sha:
        raise RuntimeError(f"Heads SHA256 mismatch vs snapshot_metadata.json: {dst_sha} != {meta_sha}")
    if expected_heads_sha256:
        req_sha = str(expected_heads_sha256).strip().upper()
        if req_sha != dst_sha:
            raise RuntimeError(f"Heads SHA256 mismatch vs --expected-heads-sha256: {dst_sha} != {req_sha}")

    if meta_path is not None and meta_path.exists():
        shutil.copyfile(meta_path, out_path / "snapshot_metadata.json")

    codebook_name: str | None = None
    codebook_sha: str | None = None
    if codebook_file:
        cb_src = pathlib.Path(codebook_file)
        if not cb_src.exists():
            raise FileNotFoundError(f"Codebook file not found: {cb_src}")
        cb_dst = out_path / cb_src.name
        shutil.copyfile(cb_src, cb_dst)
        cb_src_sha = compute_sha256(cb_src)
        cb_dst_sha = compute_sha256(cb_dst)
        if cb_src_sha != cb_dst_sha:
            raise RuntimeError(f"SHA256 mismatch copying codebook file: {cb_src_sha} != {cb_dst_sha}")
        codebook_name = cb_src.name
        codebook_sha = cb_dst_sha

    resolved_candidate = (
        candidate_name or meta.get("candidate") or (snap_path.name if snap_path is not None else out_path.name)
    )
    config_file = out_path / "config.json"
    cfg_data = json.loads(config_file.read_text(encoding="utf-8"))
    cls_cfg: dict = {
        "eval_mode": "stage2",
        "m2_k_threshold": int(m2_k_threshold),
        "m2_top_k": int(m2_top_k),
        "candidate": resolved_candidate,
        "heads_sha256": dst_sha,
    }
    if base_model_revision:
        cls_cfg["base_model_revision"] = str(base_model_revision).strip()
    if codebook_name and codebook_sha:
        cls_cfg["codebook_file"] = codebook_name
        cls_cfg["codebook_sha256"] = codebook_sha
    cfg_data["classone_config"] = cls_cfg
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
        "codebook_sha256": codebook_sha,
        "classone_config": cls_cfg,
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
    parser.add_argument(
        "--snapshot-dir", default=None, help="Stage 2 snapshot directory (optional with --allow-no-lora --heads-path)"
    )
    parser.add_argument("--heads-path", default=None, help="Explicit path to classone_heads.pt")
    parser.add_argument(
        "--expected-heads-sha256", default=None, help="Expected uppercase/lowercase SHA256 for classone_heads.pt"
    )
    parser.add_argument(
        "--allow-no-lora", action="store_true", help="Export unmodified base weights + Stage 2 readout without LoRA"
    )
    parser.add_argument(
        "--strip-multimodal", action="store_true", help="Strip unused vision/audio towers before saving"
    )
    parser.add_argument(
        "--codebook-file", default=None, help="Optional single-token codebook JSON to bundle and register"
    )
    parser.add_argument(
        "--candidate-name", default=None, help="Optional candidate identifier written into classone_config"
    )
    parser.add_argument(
        "--base-model-revision", default=None, help="Optional base model revision written into classone_config"
    )
    parser.add_argument("--output-dir", required=True, help="Destination standalone directory")
    parser.add_argument("--m2-k-threshold", type=int, default=64, help="Candidate count threshold for M2 shortlist")
    parser.add_argument("--m2-top-k", type=int, default=16, help="M2 second-pass shortlist size")
    parser.add_argument("--device", default="cpu", help="Device used while merging adapters (default: cpu)")
    args = parser.parse_args()

    export_standalone_stage2(
        base_model=args.base_model,
        snapshot_dir=args.snapshot_dir,
        output_dir=args.output_dir,
        heads_path=args.heads_path,
        expected_heads_sha256=args.expected_heads_sha256,
        allow_no_lora=args.allow_no_lora,
        strip_multimodal=args.strip_multimodal,
        codebook_file=args.codebook_file,
        candidate_name=args.candidate_name,
        base_model_revision=args.base_model_revision,
        m2_k_threshold=args.m2_k_threshold,
        m2_top_k=args.m2_top_k,
        device=args.device,
    )


if __name__ == "__main__":
    main()
