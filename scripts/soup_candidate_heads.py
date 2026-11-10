#!/usr/bin/env python3
"""Interpolates (soups) two ClassOne decision head checkpoints with proper inverse-softplus temperature clamping."""

import argparse
import os

import torch
import torch.nn.functional as F


def soup_heads(
    base_path: str,
    cand_path: str,
    alpha_choice: float,
    alpha_noul: float,
    alpha_score: float,
    out_path: str,
) -> None:
    """Linearly interpolates state dicts with head-specific alphas and proper effective temperature clamping."""
    base = torch.load(base_path, map_location="cpu")
    cand = torch.load(cand_path, map_location="cpu")

    alphas = {
        "noul_head": alpha_noul,
        "choice_head": alpha_choice,
        "score_head": alpha_score,
    }

    souped = {}
    for head_key in ["noul_head", "choice_head", "score_head"]:
        if head_key not in base or head_key not in cand:
            continue
        alpha = alphas[head_key]
        souped[head_key] = {}
        for k, v_base in base[head_key].items():
            v_cand = cand[head_key].get(k, v_base)
            if torch.is_floating_point(v_base) and v_base.shape == v_cand.shape:
                val = (1.0 - alpha) * v_base.float() + alpha * v_cand.float()
                if "temperature_raw" in k:
                    eff_t = F.softplus(val) + 0.1
                    if head_key in ("noul_head", "choice_head"):
                        clamped_t = torch.clamp(eff_t, 0.48, 0.52)
                    else:
                        clamped_t = torch.clamp(eff_t, 0.52, 0.58)
                    val = torch.log(torch.exp(clamped_t - 0.1) - 1.0 + 1e-6)
                souped[head_key][k] = val.to(v_base.dtype)
            else:
                souped[head_key][k] = v_base

    if "config" in base:
        souped["config"] = base["config"]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(souped, out_path)
    print(
        f"Saved souped heads (choice={alpha_choice:.2f}, noul={alpha_noul:.2f}, score={alpha_score:.2f}) -> {out_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Soup two ClassOne head checkpoints")
    parser.add_argument("--base", type=str, required=True, help="Baseline classone_heads.pt")
    parser.add_argument("--cand", type=str, required=True, help="Candidate classone_heads.pt")
    parser.add_argument("--alpha", type=float, default=0.65, help="Default weight on candidate (0.0 to 1.0)")
    parser.add_argument("--alpha-choice", type=float, default=None, help="Optional weight for choice_head")
    parser.add_argument("--alpha-noul", type=float, default=None, help="Optional weight for noul_head")
    parser.add_argument("--alpha-score", type=float, default=None, help="Optional weight for score_head")
    parser.add_argument("--out", type=str, required=True, help="Output classone_heads.pt")
    args = parser.parse_args()

    ac = args.alpha_choice if args.alpha_choice is not None else args.alpha
    an = args.alpha_noul if args.alpha_noul is not None else args.alpha
    as_ = args.alpha_score if args.alpha_score is not None else args.alpha
    soup_heads(args.base, args.cand, ac, an, as_, args.out)


if __name__ == "__main__":
    main()
