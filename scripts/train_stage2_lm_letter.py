#!/usr/bin/env python3
"""Stage 2 Multi-GPU Trainer: Supports Arm A (Raw Qwen + Chat Template + LoRA r=32) & Arm B (ClassOne + LoRA r=16).

- Arm A (primary): Raw Qwen/Qwen3.5-4B (chat template), fresh LoRA r=32, LMHeadLetterReadout primary (scale=1.0)
  + auxiliary FP32 ChoiceHead/NoulHead/ScoreHead (weight=0.25 on Choice), heads warm-started from classone_heads.pt
  where shapes match.
- Arm B (as piloted): devops-thiago/classone-qwen3.5-4b + LoRA r=16, ClassOnePromptBuilder(options_in_prefix=False)
  with 2D position-invariant masking, LMHeadLetterReadout primary + auxiliary heads.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoTokenizer

from classone.modeling.loss import RLCDLoss
from classone.modeling.modeling_classone import ClassOneModel
from classone.schemas import ChoiceQuestion, NoulQuestion, ScoreQuestion
from classone.stage2_readout import (
    ChatDecisionPromptBuilder,
    compute_lm_letter_logits,
    load_codebook_lm_head,
    strip_unused_multimodal_towers,
)
from classone.tokenizer import ClassOnePromptBuilder
from classone.train.trainer import ClassOneTrainer


def build_label_token_map(tokenizer) -> tuple[list[str], dict[str, int]]:
    """Maps single-token option labels (A..Z, a..z, 0..9, boolean/NLI labels) to vocabulary token IDs."""
    candidates = (
        [chr(ord("A") + i) for i in range(26)]
        + [chr(ord("a") + i) for i in range(26)]
        + [str(i) for i in range(10)]
        + ["Yes", "No", "yes", "no", "true", "false", "Entailment", "Contradiction", "legitimate", "malicious"]
    )
    ordered_letters = []
    tok_map = {}
    for c in candidates:
        ids = tokenizer.encode(c, add_special_tokens=False)
        if ids:
            tok_map[c] = ids[0]
            if len(c) == 1 and c.isalpha() and c.isupper():
                ordered_letters.append(c)
    return ordered_letters, tok_map


def parse_item_questions(raw_qs: dict) -> dict:
    """Converts serialized question dicts into ClassOne Question schema objects."""
    parsed = {}
    for qid, q_obj in raw_qs.items():
        q_type = q_obj.get("type", "choice")
        if q_type == "noul":
            parsed[qid] = NoulQuestion(instructions=q_obj.get("instructions", ""))
        elif q_type == "choice":
            parsed[qid] = ChoiceQuestion(
                instructions=q_obj.get("instructions", ""),
                criteria=q_obj.get("criteria", {}),
            )
        elif q_type == "score":
            parsed[qid] = ScoreQuestion(
                instructions=q_obj.get("instructions", ""),
                criteria=q_obj.get("criteria", ["1", "2", "3"]),
            )
    return parsed


def load_heads_matching_shapes(model: ClassOneModel, heads_path: str, device: torch.device, rank: int) -> int:
    """Loads parameters from classone_heads.pt wherever tensor shapes match exactly."""
    heads = torch.load(heads_path, map_location=device, weights_only=True)
    loaded_tensors = 0
    for attr_name in ("noul_head", "choice_head", "score_head"):
        submod = getattr(model, attr_name, None)
        src_dict = heads.get(attr_name, {})
        if submod is None or not isinstance(src_dict, dict):
            continue
        dst_dict = submod.state_dict()
        filtered = {k: v for k, v in src_dict.items() if k in dst_dict and dst_dict[k].shape == v.shape}
        submod.load_state_dict(filtered, strict=False)
        loaded_tensors += len(filtered)
    if rank == 0:
        print(f"[Stage2 Rank {rank}] Loaded {loaded_tensors} shape-matched head tensors from {heads_path}", flush=True)
    return loaded_tensors


def _resolve_target_idx(span, target: int | float | str) -> int | None:
    if isinstance(target, str):
        if target in span.option_keys:
            return span.option_keys.index(target)
        return None
    idx = int(target)
    if 0 <= idx < len(span.option_keys):
        return idx
    return None


def _build_step_units_arm_a(
    raw_batch: list[dict],
    chat_builder: ChatDecisionPromptBuilder,
    epoch_idx: int,
    max_seq_len: int,
) -> tuple[list[tuple], int]:
    """Converts a batch of multi-task requests into Arm A chat-templated forward units."""
    units = []
    total_valid_q = 0
    for item_idx, raw in enumerate(raw_batch):
        qs = parse_item_questions(raw["questions"])
        tgts = raw.get("targets", raw.get("gold", {}))
        qids = [qid for qid in qs if qid in tgts]
        if not qids:
            continue

        # Preserve full supervision on JevBench replay items; rotate 2 q/req per epoch on large multi-q sets
        is_replay = raw.get("source") == "ClassOne_JevBench_Replay"
        if len(qids) > 2 and not is_replay:
            start = (epoch_idx * 2 + item_idx) % len(qids)
            selected_qids = [qids[(start + j) % len(qids)] for j in range(2)]
        else:
            selected_qids = qids

        for qid in selected_qids:
            q_obj = qs[qid]
            packed = chat_builder.pack_question(state=raw["state"], q_id=qid, question=q_obj)
            if packed.input_ids.shape[1] > max_seq_len:
                state_str = chat_builder.serialize_state(raw["state"])
                packed = chat_builder.pack_question(state=state_str[: max_seq_len * 2], q_id=qid, question=q_obj)
            units.append((packed, tgts[qid], False, packed.label_token_ids[qid]))
            total_valid_q += 1
    return units, total_valid_q


def _build_step_units_arm_b(
    raw_batch: list[dict],
    co_builder: ClassOnePromptBuilder,
    max_seq_len: int,
) -> tuple[list[tuple], int]:
    """Converts a batch of requests into Arm B packed sequences (options_in_prefix=False)."""
    units = []
    total_valid_q = 0
    for raw in raw_batch:
        qs = parse_item_questions(raw["questions"])
        tgts = raw.get("targets", raw.get("gold", {}))
        packed = co_builder.pack(state=raw["state"], questions=qs, options_in_prefix=False)
        if packed.input_ids.shape[1] > max_seq_len:
            state_str = co_builder.serialize_state(raw["state"])
            packed = co_builder.pack(state=state_str[: max_seq_len * 2], questions=qs, options_in_prefix=False)
        valid_here = sum(1 for qid in tgts if qid in packed.questions)
        if valid_here > 0:
            units.append((packed, tgts, True, None))
            total_valid_q += valid_here
    return units, total_valid_q


def _compute_unit_losses(
    model: ClassOneModel,
    packed,
    tgts_or_single,
    use_pos_inv: bool,
    explicit_label_ids: list[int] | None,
    device: torch.device,
    tokenizer,
    ordered_letters: list[str],
    tok_map: dict[str, int],
    embed_weight: torch.Tensor,
    loss_fn_lm: RLCDLoss,
    loss_fn_heads: RLCDLoss,
    lm_scale: float,
    aux_head_weight: float,
    kl_weight: float = 0.0,
    input_device: torch.device | None = None,
) -> tuple[torch.Tensor | None, float, float]:
    in_dev = input_device or device
    input_ids = packed.input_ids.to(in_dev)
    attention_mask = packed.attention_mask.to(in_dev)

    has_opts = any(len(getattr(sp, "option_token_spans", [])) > 1 for sp in packed.questions.values())
    if use_pos_inv and has_opts:
        mask_4d, pos_ids = model.build_position_invariant_mask(packed, in_dev, dtype=torch.bfloat16)
        hidden_states = model.extract_hidden_states(input_ids=input_ids, attention_mask=mask_4d, position_ids=pos_ids)
    else:
        hidden_states = model.extract_hidden_states(input_ids=input_ids, attention_mask=attention_mask)

    ref_seq_hidden = None
    if kl_weight > 0.0 and hasattr(model.backbone, "disable_adapter"):
        with torch.no_grad(), model.backbone.disable_adapter():
            if use_pos_inv and has_opts:
                ref_hs = model.extract_hidden_states(input_ids=input_ids, attention_mask=mask_4d, position_ids=pos_ids)
            else:
                ref_hs = model.extract_hidden_states(input_ids=input_ids, attention_mask=attention_mask)
            ref_seq_hidden = ref_hs[0].detach()
            del ref_hs

    seq_hidden = hidden_states[0]
    head_dtype = next(model.noul_head.parameters()).dtype
    seq_hidden_fp32 = seq_hidden.to(head_dtype)

    item_loss = torch.tensor(0.0, device=device)
    item_lm_val = 0.0
    item_hd_val = 0.0
    item_q_cnt = 0

    tgt_map = (
        tgts_or_single if isinstance(tgts_or_single, dict) else {next(iter(packed.questions.keys())): tgts_or_single}
    )

    for qid, target in tgt_map.items():
        if qid not in packed.questions:
            continue
        span = packed.questions[qid]
        h_q_fp32 = seq_hidden_fp32[span.query_token_idx]
        h_q_bf16 = seq_hidden[span.query_token_idx]

        if span.question_type == "noul":
            prob, logit = model.noul_head(h_q_fp32, return_logits=True)
            t_float = float(target)
            t_tensor = torch.tensor([t_float], device=device, dtype=torch.float32)
            q_hd_loss = loss_fn_heads.forward_binary(prob, t_tensor, logit=logit)

            if explicit_label_ids is not None and len(explicit_label_ids) == 2:
                probs_lm, logits_lm = compute_lm_letter_logits(
                    h_q_bf16, embed_weight, explicit_label_ids, scale=lm_scale
                )
                lm_idx = 0 if t_float >= 0.5 else 1
                t_lm = torch.tensor([lm_idx], device=device, dtype=torch.long)
                q_lm_loss = loss_fn_lm.forward_multiclass(probs_lm.unsqueeze(0), t_lm, logits=logits_lm.unsqueeze(0))
                if ref_seq_hidden is not None:
                    with torch.no_grad():
                        p_ref, _ = compute_lm_letter_logits(
                            ref_seq_hidden[span.query_token_idx], embed_weight, explicit_label_ids, scale=lm_scale
                        )
                    q_kl = F.kl_div(
                        F.log_softmax(logits_lm, dim=-1).unsqueeze(0), p_ref.unsqueeze(0), reduction="batchmean"
                    )
                    q_lm_loss = q_lm_loss + kl_weight * q_kl
                item_loss = item_loss + (1.0 * q_lm_loss + 1.0 * q_hd_loss)
                item_lm_val += float(q_lm_loss.item())
                item_hd_val += float(q_hd_loss.item())
            else:
                item_loss = item_loss + q_hd_loss
                item_hd_val += float(q_hd_loss.item())
            item_q_cnt += 1

        elif span.question_type == "choice":
            target_idx = _resolve_target_idx(span, target)
            if target_idx is None:
                continue
            t_tensor = torch.tensor([target_idx], device=device, dtype=torch.long)

            h_opts = torch.stack([seq_hidden_fp32[idx] for idx in span.option_token_indices])
            probs_ch, logits_ch = model.choice_head(h_q_fp32, h_opts, return_logits=True)
            q_hd_loss = loss_fn_heads.forward_multiclass(probs_ch.unsqueeze(0), t_tensor, logits=logits_ch.unsqueeze(0))

            if explicit_label_ids is not None:
                cand_tok_ids = explicit_label_ids
            else:
                cand_tok_ids = []
                for o_i, o_key in enumerate(span.option_keys):
                    if o_key in tok_map:
                        cand_tok_ids.append(tok_map[o_key])
                    else:
                        ids_k = tokenizer.encode(str(o_key).strip(), add_special_tokens=False)
                        cand_tok_ids.append(ids_k[0] if ids_k else tok_map[ordered_letters[o_i % len(ordered_letters)]])

            probs_lm, logits_lm = compute_lm_letter_logits(h_q_bf16, embed_weight, cand_tok_ids, scale=lm_scale)
            q_lm_loss = loss_fn_lm.forward_multiclass(probs_lm.unsqueeze(0), t_tensor, logits=logits_lm.unsqueeze(0))
            if ref_seq_hidden is not None:
                with torch.no_grad():
                    p_ref, _ = compute_lm_letter_logits(
                        ref_seq_hidden[span.query_token_idx], embed_weight, cand_tok_ids, scale=lm_scale
                    )
                q_kl = F.kl_div(
                    F.log_softmax(logits_lm, dim=-1).unsqueeze(0), p_ref.unsqueeze(0), reduction="batchmean"
                )
                q_lm_loss = q_lm_loss + kl_weight * q_kl

            item_loss = item_loss + (1.0 * q_lm_loss + aux_head_weight * q_hd_loss)
            item_lm_val += float(q_lm_loss.item())
            item_hd_val += float(q_hd_loss.item())
            item_q_cnt += 1

        elif span.question_type == "score":
            h_levels = torch.stack([seq_hidden_fp32[idx] for idx in span.option_token_indices])
            _, probs_sc, logits_sc = model.score_head(h_q_fp32, h_levels, return_logits=True)
            if isinstance(target, (int, float)):
                target_idx = max(0, min(len(span.option_keys) - 1, int(target) - 1))
            elif str(target) in span.option_keys:
                target_idx = span.option_keys.index(str(target))
            else:
                target_idx = 0
            t_tensor = torch.tensor([target_idx], device=device, dtype=torch.long)
            q_hd_loss = loss_fn_heads.forward_multiclass(probs_sc.unsqueeze(0), t_tensor, logits=logits_sc.unsqueeze(0))

            if explicit_label_ids is not None:
                probs_lm, logits_lm = compute_lm_letter_logits(
                    h_q_bf16, embed_weight, explicit_label_ids, scale=lm_scale
                )
                q_lm_loss = loss_fn_lm.forward_multiclass(
                    probs_lm.unsqueeze(0), t_tensor, logits=logits_lm.unsqueeze(0)
                )
                if ref_seq_hidden is not None:
                    with torch.no_grad():
                        p_ref, _ = compute_lm_letter_logits(
                            ref_seq_hidden[span.query_token_idx], embed_weight, explicit_label_ids, scale=lm_scale
                        )
                    q_kl = F.kl_div(
                        F.log_softmax(logits_lm, dim=-1).unsqueeze(0), p_ref.unsqueeze(0), reduction="batchmean"
                    )
                    q_lm_loss = q_lm_loss + kl_weight * q_kl
                item_loss = item_loss + (1.0 * q_lm_loss + 1.0 * q_hd_loss)
                item_lm_val += float(q_lm_loss.item())
                item_hd_val += float(q_hd_loss.item())
            else:
                item_loss = item_loss + q_hd_loss
                item_hd_val += float(q_hd_loss.item())
            item_q_cnt += 1

    del seq_hidden, seq_hidden_fp32, hidden_states, ref_seq_hidden
    if item_q_cnt == 0:
        return None, 0.0, 0.0
    return item_loss, item_lm_val, item_hd_val


def _eval_mcq5_and_oos(
    model: ClassOneModel,
    chat_builder: ChatDecisionPromptBuilder,
    embed_weight: torch.Tensor,
    mcq5_rows: list[dict],
    oos_rows: list[dict],
    device: torch.device,
    lm_scale: float,
    input_device: torch.device | None = None,
) -> dict:
    """Evaluates Held-Out-1K single_q_knowledge_mcq5 (120) and Out-of-Source-320 (320) on rank 0."""
    model.eval()
    in_dev = input_device or device
    t0 = time.perf_counter()

    def _run_set(rows: list[dict]) -> tuple[float, dict[str, float]]:
        ok_cnt = 0
        q_cnt = 0
        by_slice: dict[str, list[int]] = {}
        for r in rows:
            sl = r.get("slice", "default")
            tgts = r.get("gold", r.get("targets", {}))
            for qid, qd in r["questions"].items():
                if qid not in tgts:
                    continue
                q_obj = ChoiceQuestion(instructions=qd.get("instructions", ""), criteria=qd.get("criteria", {}))
                packed = chat_builder.pack_question(state=r["state"], q_id=qid, question=q_obj)
                if packed.input_ids.shape[1] > 4096:
                    packed = chat_builder.pack_question(
                        state=chat_builder.serialize_state(r["state"])[:8000], q_id=qid, question=q_obj
                    )
                with torch.no_grad():
                    hs = model.extract_hidden_states(packed.input_ids.to(in_dev), packed.attention_mask.to(in_dev))
                    span = packed.questions[qid]
                    _, logits_lm = compute_lm_letter_logits(
                        hs[0][span.query_token_idx], embed_weight, packed.label_token_ids[qid], scale=lm_scale
                    )
                pred_k = span.option_keys[int(torch.argmax(logits_lm).item())]
                hit = int(str(pred_k) == str(tgts[qid]))
                ok_cnt += hit
                q_cnt += 1
                by_slice.setdefault(sl, []).append(hit)
        per_sl = {k: round(100.0 * sum(v) / len(v), 2) for k, v in by_slice.items() if v}
        return round(100.0 * ok_cnt / max(1, q_cnt), 2), per_sl

    mcq5_acc, _ = _run_set(mcq5_rows) if mcq5_rows else (0.0, {})
    oos_acc, oos_slices = _run_set(oos_rows) if oos_rows else (0.0, {})
    torch.cuda.empty_cache()
    return {
        "mcq5_acc": mcq5_acc,
        "oos_320_acc": oos_acc,
        "oos_slices": oos_slices,
        "eval_sec": round(time.perf_counter() - t0, 2),
    }


def run_stage2_worker(
    rank: int,
    world_size: int,
    args: argparse.Namespace,
    q_out=None,
    q_in=None,
    ctrl_q=None,
):
    """Executes Stage 2 training (Arm A or Arm B) on a single GPU worker or 2-GPU model parallel."""
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    is_mp = bool(getattr(args, "model_parallel", False))
    if is_mp:
        if torch.cuda.device_count() < 2:
            raise RuntimeError(f"--model-parallel requires >= 2 CUDA devices, found {torch.cuda.device_count()}")
        input_device = torch.device("cuda:0")
        output_device = torch.device("cuda:1")
        device = output_device
        torch.cuda.set_device(input_device)
        torch.cuda.reset_peak_memory_stats(input_device)
        torch.cuda.reset_peak_memory_stats(output_device)
    else:
        device = torch.device(f"cuda:{rank}")
        input_device = device
        output_device = device
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    if rank == 0:
        mp_note = " | model_parallel=cuda:0->cuda:1" if is_mp else ""
        print(
            f"[Stage2 Arm {args.arm} | Rank {rank}{mp_note}] Loading {args.base_model} "
            f"(LoRA r={args.lora_r}, alpha={args.lora_alpha}, lm_scale={args.lm_scale})...",
            flush=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, local_files_only=True)
    chat_builder = ChatDecisionPromptBuilder(tokenizer) if args.arm.upper() == "A" else None
    co_builder = ClassOnePromptBuilder(tokenizer)

    if is_mp:
        model = ClassOneModel.from_backbone(
            args.base_model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            model_parallel=True,
            mp_devices=("cuda:0", "cuda:1"),
            mp_split_layer=args.mp_split_layer,
            head_dtype=torch.float32,
        )
    else:
        model = ClassOneModel.from_backbone(
            args.base_model,
            tokenizer=tokenizer,
            device=str(device),
            torch_dtype=torch.bfloat16,
        )

    heads_path = args.pretrained_heads or os.path.join(args.base_model, "classone_heads.pt")
    if os.path.exists(heads_path):
        load_heads_matching_shapes(model, heads_path, output_device, rank)

    if chat_builder is not None:
        codebook_head = load_codebook_lm_head(
            args.base_model,
            chat_builder,
            device=output_device,
            dtype=torch.bfloat16,
            include_yes_no=True,
        )
        if codebook_head is not None:
            model.lm_readout_head = codebook_head
            if rank == 0:
                print(
                    f"[Stage2 Arm {args.arm}] Loaded frozen CodebookLMHead slice "
                    f"{tuple(codebook_head.weight.shape)} on {output_device}"
                    + (
                        f" (softcap={codebook_head.final_logit_softcapping})"
                        if codebook_head.final_logit_softcapping is not None
                        else ""
                    ),
                    flush=True,
                )

    stripped_towers = strip_unused_multimodal_towers(model.backbone, clear_cuda_cache=True)
    if stripped_towers and rank == 0:
        print(f"[Stage2 Arm {args.arm}] Stripped unused multimodal towers: {stripped_towers}", flush=True)

    if hasattr(model.backbone, "gradient_checkpointing_enable"):
        model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    loss_fn_lm = RLCDLoss(
        brier_weight=0.25,
        log_weight=1.0,
        margin_weight=0.25,
        margin_threshold=0.8,
        focal_gamma=args.lm_focal_gamma,
    )
    loss_fn_heads = RLCDLoss(
        brier_weight=0.5,
        log_weight=1.0,
        margin_weight=0.5,
        margin_threshold=1.0,
        focal_gamma=args.head_focal_gamma,
    )

    trainer = ClassOneTrainer(
        model=model,
        prompt_builder=co_builder,
        loss_fn=loss_fn_heads,
        lr=args.lr,
        weight_decay=0.01,
        device=output_device,
    )
    if args.resume_lora and os.path.exists(args.resume_lora):
        model.backbone = PeftModel.from_pretrained(model.backbone, args.resume_lora, is_trainable=True)
        if is_mp:
            for mod in model.backbone.modules():
                base_layer = getattr(mod, "base_layer", None)
                base_w = getattr(base_layer, "weight", None)
                if base_w is None:
                    continue
                target_dev = base_w.device
                for attr in ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B", "lora_magnitude_vector"):
                    sub = getattr(mod, attr, None)
                    if isinstance(sub, torch.nn.Module):
                        sub.to(device=target_dev)
        for head in (model.noul_head, model.choice_head, model.score_head):
            for p in head.parameters():
                p.requires_grad = True
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        trainer.optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
        trainer.scheduler = None
        if rank == 0:
            print(
                f"[Stage2 Arm {args.arm} | Rank {rank}] Resumed trainable LoRA adapter from {args.resume_lora} at lr={args.lr}",
                flush=True,
            )
    else:
        trainer.enable_lora(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05)

    if is_mp:
        assert all(p.device.type == "cuda" for p in model.parameters() if p.requires_grad), (
            "Model-parallel mode requires all trainable parameters on CUDA devices"
        )

    if hasattr(model.backbone, "enable_input_require_grads"):
        model.backbone.enable_input_require_grads()

    mcq5_rows: list[dict] = []
    oos_rows: list[dict] = []
    if rank == 0 and args.eval_every > 0:
        if args.eval_heldout1k_path and os.path.exists(args.eval_heldout1k_path):
            mcq5_rows = [
                json.loads(line)
                for line in Path(args.eval_heldout1k_path).read_text(encoding="utf-8").splitlines()
                if line.strip() and json.loads(line).get("slice") == "single_q_knowledge_mcq5"
            ]
        if args.eval_oos_path and os.path.exists(args.eval_oos_path):
            oos_rows = [
                json.loads(line)
                for line in Path(args.eval_oos_path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        print(
            f"[Stage2 EvalMonitor] Loaded mcq5={len(mcq5_rows)} rows, oos320={len(oos_rows)} rows (eval every {args.eval_every} steps)",
            flush=True,
        )

    all_lines = Path(args.dataset_path).read_text(encoding="utf-8").splitlines()
    raw_items = [json.loads(line) for line in all_lines if line.strip()]
    rank_items = [raw_items[i] for i in range(rank, len(raw_items), world_size)]

    items_per_step = args.batch_size * args.gradient_accumulation_steps
    # Compute global steps_per_epoch symmetrically so every rank executes the exact same step count.
    max_rank_items = (len(raw_items) + world_size - 1) // world_size
    steps_per_epoch = max(1, (max_rank_items + items_per_step - 1) // items_per_step)
    total_steps = args.max_steps if args.max_steps is not None else (steps_per_epoch * args.epochs)
    trainer.setup_scheduler(total_steps=total_steps, warmup_steps=max(5, min(100, total_steps // 12)))

    ordered_letters, tok_map = build_label_token_map(tokenizer)
    embed_weight = (
        model.lm_readout_head
        if getattr(model, "lm_readout_head", None) is not None
        else model.backbone.get_input_embeddings().weight
    )

    step_logs = []
    eval_history = []
    best_mcq5 = 0.0
    t_start = time.perf_counter()

    for step_idx in range(1, total_steps + 1):
        t_step = time.perf_counter()
        epoch_idx = (step_idx - 1) // steps_per_epoch
        trainer.model.train()
        trainer.optimizer.zero_grad(set_to_none=True)

        start_item_idx = ((step_idx - 1) % steps_per_epoch) * items_per_step
        batch_slice = [rank_items[(start_item_idx + j) % len(rank_items)] for j in range(items_per_step)]

        if args.arm.upper() == "A":
            assert chat_builder is not None
            units, total_valid_q = _build_step_units_arm_a(batch_slice, chat_builder, epoch_idx, args.max_seq_len)
        else:
            units, total_valid_q = _build_step_units_arm_b(batch_slice, co_builder, args.max_seq_len)

        total_valid_q = max(1, total_valid_q)
        seq_lens = [int(u[0].input_ids.shape[1]) for u in units] or [0]
        step_total_loss = 0.0
        step_lm_loss = 0.0
        step_head_loss = 0.0

        for packed, tgts_or_single, use_pos_inv, explicit_label_ids in units:
            item_loss, item_lm_val, item_hd_val = _compute_unit_losses(
                model=model,
                packed=packed,
                tgts_or_single=tgts_or_single,
                use_pos_inv=use_pos_inv,
                explicit_label_ids=explicit_label_ids,
                device=output_device,
                tokenizer=tokenizer,
                ordered_letters=ordered_letters,
                tok_map=tok_map,
                embed_weight=embed_weight,
                loss_fn_lm=loss_fn_lm,
                loss_fn_heads=loss_fn_heads,
                lm_scale=args.lm_scale,
                aux_head_weight=args.aux_head_weight,
                kl_weight=args.kl_weight,
                input_device=input_device,
            )
            if item_loss is not None:
                scaled = item_loss / total_valid_q
                scaled.backward()
                step_total_loss += float(item_loss.item()) / total_valid_q
                step_lm_loss += item_lm_val / total_valid_q
                step_head_loss += item_hd_val / total_valid_q
                del item_loss

        if step_idx == 1:
            if is_mp:
                g0_nz = sum(
                    1
                    for p in model.parameters()
                    if p.requires_grad and p.device == input_device and p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                )
                g1_nz = sum(
                    1
                    for p in model.parameters()
                    if p.requires_grad and p.device == output_device and p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                )
                if g0_nz == 0 or g1_nz == 0:
                    raise RuntimeError(f"Model-parallel gradient check failed at step 1: {input_device}={g0_nz}, {output_device}={g1_nz}")
                print(f"[MP Step 1 Grads Verified] {input_device}_nonzero={g0_nz}, {output_device}_nonzero={g1_nz}", flush=True)
            else:
                lora_nz = sum(
                    1
                    for n, p in model.named_parameters()
                    if "lora_" in n and p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                )
                head_nz = sum(
                    1
                    for n, p in model.named_parameters()
                    if any(h in n for h in ("noul_head.", "choice_head.", "score_head."))
                    and p.requires_grad
                    and p.grad is not None
                    and torch.isfinite(p.grad).all()
                    and p.grad.abs().sum() > 0
                )
                if lora_nz == 0 or head_nz == 0:
                    raise RuntimeError(
                        f"Step 1 gradient check failed on rank {rank}: lora_nonzero={lora_nz}, head_nonzero={head_nz}"
                    )
                if rank == 0:
                    print(f"[Step 1 Grads Verified] lora_nonzero={lora_nz}, head_nonzero={head_nz}", flush=True)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if world_size > 1 and q_out is not None and q_in is not None:
            trainer.sync_gradients_shm(q_out, q_in)
        trainer.optimizer.step()
        if trainer.scheduler is not None:
            trainer.scheduler.step()
        trainer.optimizer.zero_grad(set_to_none=True)

        if step_idx % 20 == 0:
            if is_mp:
                for dev_i in (input_device, output_device):
                    torch.cuda.synchronize(dev_i)
                torch.cuda.empty_cache()
            else:
                torch.cuda.empty_cache()

        step_sec = time.perf_counter() - t_step
        if is_mp:
            g0_alloc = torch.cuda.memory_allocated(input_device) / 1e9
            g0_max_alloc = torch.cuda.max_memory_allocated(input_device) / 1e9
            g0_res = torch.cuda.memory_reserved(input_device) / 1e9
            g0_max_res = torch.cuda.max_memory_reserved(input_device) / 1e9
            g1_alloc = torch.cuda.memory_allocated(output_device) / 1e9
            g1_max_alloc = torch.cuda.max_memory_allocated(output_device) / 1e9
            g1_res = torch.cuda.memory_reserved(output_device) / 1e9
            g1_max_res = torch.cuda.max_memory_reserved(output_device) / 1e9
            alloc_gb = max(g0_alloc, g1_alloc)
            max_alloc_gb = max(g0_max_alloc, g1_max_alloc)
            res_gb = max(g0_max_res, g1_max_res)
            rec = {
                "step": step_idx,
                "epoch": epoch_idx + 1,
                "step_sec": round(step_sec, 3),
                "total_loss": round(step_total_loss, 4),
                "lm_letter_loss": round(step_lm_loss, 4),
                "aux_head_loss": round(step_head_loss, 4),
                "alloc_gb": round(alloc_gb, 3),
                "max_alloc_gb": round(max_alloc_gb, 3),
                "reserved_gb": round(res_gb, 3),
                "gpu0_alloc_gb": round(g0_alloc, 3),
                "gpu0_max_alloc_gb": round(g0_max_alloc, 3),
                "gpu0_reserved_gb": round(g0_res, 3),
                "gpu0_max_reserved_gb": round(g0_max_res, 3),
                "gpu1_alloc_gb": round(g1_alloc, 3),
                "gpu1_max_alloc_gb": round(g1_max_alloc, 3),
                "gpu1_reserved_gb": round(g1_res, 3),
                "gpu1_max_reserved_gb": round(g1_max_res, 3),
                "avg_seq_tokens": round(statistics.mean(seq_lens), 1),
            }
        else:
            alloc_gb = torch.cuda.memory_allocated(device) / 1e9
            max_alloc_gb = torch.cuda.max_memory_allocated(device) / 1e9
            res_gb = torch.cuda.memory_reserved(device) / 1e9
            rec = {
                "step": step_idx,
                "epoch": epoch_idx + 1,
                "step_sec": round(step_sec, 3),
                "total_loss": round(step_total_loss, 4),
                "lm_letter_loss": round(step_lm_loss, 4),
                "aux_head_loss": round(step_head_loss, 4),
                "alloc_gb": round(alloc_gb, 3),
                "max_alloc_gb": round(max_alloc_gb, 3),
                "reserved_gb": round(res_gb, 3),
                "avg_seq_tokens": round(statistics.mean(seq_lens), 1),
            }
        step_logs.append(rec)

        log_every = 5 if args.max_steps and args.max_steps <= 100 else 25
        if rank == 0 and (step_idx == 1 or step_idx % log_every == 0 or step_idx == total_steps):
            elapsed = time.perf_counter() - t_start
            if is_mp:
                vram_msg = (
                    f"VRAM peak=({rec['gpu0_max_alloc_gb']:.2f}, {rec['gpu1_max_alloc_gb']:.2f})GB "
                    f"res=({rec['gpu0_max_reserved_gb']:.2f}, {rec['gpu1_max_reserved_gb']:.2f})GB"
                )
            else:
                vram_msg = f"VRAM peak={max_alloc_gb:.2f}GB res={res_gb:.2f}GB"
            print(
                f"[Arm {args.arm} Step {step_idx:4d}/{total_steps}] {step_sec:.2f}s/step (avg {elapsed / step_idx:.2f}s) | "
                f"loss={step_total_loss:.4f} (lm={step_lm_loss:.4f}, head={step_head_loss:.4f}) | "
                f"{vram_msg} | tok={rec['avg_seq_tokens']}",
                flush=True,
            )

        if args.max_vram_gb is not None:
            peak_check = max(rec.get("gpu0_max_alloc_gb", max_alloc_gb), rec.get("gpu1_max_alloc_gb", max_alloc_gb))
            if peak_check > args.max_vram_gb:
                if args.output_dir:
                    os.makedirs(args.output_dir, exist_ok=True)
                    Path(args.output_dir, f"telemetry_rank{rank}.json").write_text(
                        json.dumps(step_logs, indent=2), encoding="utf-8"
                    )
                raise RuntimeError(
                    f"VRAM ceiling exceeded at step {step_idx}: peak allocated {peak_check:.3f}GB > {args.max_vram_gb}GB"
                )

        # Save intermediate epoch checkpoints for full runs
        if rank == 0 and args.output_dir and args.max_steps is None and (step_idx % steps_per_epoch == 0):
            ep_dir = os.path.join(args.output_dir, f"checkpoint-epoch{epoch_idx + 1}")
            os.makedirs(ep_dir, exist_ok=True)
            trainer.save_checkpoint(ep_dir)
            print(f"[Arm {args.arm}] Saved Epoch {epoch_idx + 1} checkpoint -> {ep_dir}", flush=True)

        if args.eval_every > 0 and (step_idx % args.eval_every == 0 or step_idx == total_steps):
            stop_now = False
            if rank == 0 and chat_builder is not None:
                ev = _eval_mcq5_and_oos(
                    model=model,
                    chat_builder=chat_builder,
                    embed_weight=embed_weight,
                    mcq5_rows=mcq5_rows,
                    oos_rows=oos_rows,
                    device=output_device,
                    lm_scale=args.lm_scale,
                    input_device=input_device,
                )
                ev["step"] = step_idx
                eval_history.append(ev)
                best_mcq5 = max(best_mcq5, ev["mcq5_acc"])
                if args.output_dir:
                    step_dir = os.path.join(args.output_dir, f"checkpoint-step{step_idx}")
                    os.makedirs(step_dir, exist_ok=True)
                    trainer.save_checkpoint(step_dir)
                    Path(args.output_dir, "eval_checkpoints.json").write_text(
                        json.dumps(eval_history, indent=2), encoding="utf-8"
                    )
                print(
                    f"[Arm {args.arm} Step {step_idx:4d} Eval] mcq5={ev['mcq5_acc']:.2f}% (best={best_mcq5:.2f}%) | "
                    f"oos320={ev['oos_320_acc']:.2f}% {ev['oos_slices']} ({ev['eval_sec']:.1f}s)",
                    flush=True,
                )
                if (
                    (not args.no_early_stop)
                    and step_idx >= 400
                    and (
                        (oos_rows and ev["oos_320_acc"] < args.oos_baseline_acc - 1.5)
                        or (best_mcq5 >= 72.0 and ev["mcq5_acc"] < best_mcq5 - 4.0)
                    )
                ):
                    stop_now = True
                    print(
                        f"[Arm {args.arm} Step {step_idx:4d}] Early stopping triggered on forgetting threshold!",
                        flush=True,
                    )
            if world_size > 1 and ctrl_q is not None:
                if rank == 0:
                    ctrl_q.put(stop_now)
                else:
                    stop_now = bool(ctrl_q.get())
            if stop_now:
                break

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        Path(args.output_dir, f"telemetry_rank{rank}.json").write_text(
            json.dumps(step_logs, indent=2), encoding="utf-8"
        )
        if rank == 0:
            trainer.save_checkpoint(args.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2 LM-Head Letter Readout + Auxiliary Head Trainer")
    parser.add_argument(
        "--arm", type=str, default="A", choices=["A", "B"], help="Training arm: A (raw chat r32) or B (classone r16)"
    )
    parser.add_argument("--base-model", type=str, required=True, help="Base model path or HF repo ID")
    parser.add_argument("--pretrained-heads", type=str, default=None, help="Path to classone_heads.pt")
    parser.add_argument("--resume-lora", type=str, default=None, help="Optional LoRA adapter dir to resume training")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to 3-layer filtered JSONL corpus")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for checkpoints & telemetry")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Optional cap on optimizer steps (e.g. 60-step pilot)"
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Micro-batch size per GPU")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--max-seq-len", type=int, default=1536, help="Max token length per training sequence")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank (32 for Arm A, 16 for Arm B)")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha (64 for Arm A, 32 for Arm B)")
    parser.add_argument("--lm-scale", type=float, default=1.0, help="Logit scaling factor for LMHeadLetterReadout")
    parser.add_argument("--lm-focal-gamma", type=float, default=1.5, help="Focal gamma for LM-head letter loss")
    parser.add_argument("--head-focal-gamma", type=float, default=2.0, help="Focal gamma for auxiliary decision heads")
    parser.add_argument("--aux-head-weight", type=float, default=0.25, help="Weight on auxiliary ChoiceHead loss")
    parser.add_argument(
        "--kl-weight", type=float, default=0.0, help="Anti-forgetting KL weight toward frozen base readout"
    )
    parser.add_argument("--gpus", type=int, default=2, help="Number of local GPUs")
    parser.add_argument("--eval-every", type=int, default=0, help="Evaluate mcq5 + oos320 every N steps on rank 0")
    parser.add_argument(
        "--no-early-stop", action="store_true", help="Log + checkpoint every eval-every steps without halting early"
    )
    parser.add_argument("--eval-heldout1k-path", type=str, default=None, help="Path to heldout-non-suite-1k.jsonl")
    parser.add_argument("--eval-oos-path", type=str, default=None, help="Path to heldout-out-of-source-300.jsonl")
    parser.add_argument(
        "--oos-baseline-acc", type=float, default=83.12, help="Arm A OOS-320 baseline for early stopping"
    )
    parser.add_argument(
        "--model-parallel",
        action="store_true",
        help="Shard transformer layers 50/50 across cuda:0 and cuda:1 with heads/readout on cuda:1",
    )
    parser.add_argument(
        "--mp-split-layer",
        type=int,
        default=None,
        help="Optional split layer index for --model-parallel (default: len(layers)//2)",
    )
    parser.add_argument(
        "--max-vram-gb",
        type=float,
        default=None,
        help="Optional per-GPU peak allocated VRAM safety limit in GB",
    )
    return parser.parse_args()


def summarize_telemetry(out_dir: str, arm: str, world_size: int) -> dict:
    logs0 = json.loads(Path(out_dir, "telemetry_rank0.json").read_text(encoding="utf-8"))
    logs1 = json.loads(Path(out_dir, "telemetry_rank1.json").read_text(encoding="utf-8")) if world_size > 1 else logs0
    steady = [x["step_sec"] for x in logs0[2:]] if len(logs0) > 2 else [x["step_sec"] for x in logs0]
    buckets = []
    bucket_sz = 10 if len(logs0) <= 100 else max(1, len(logs0) // 6)
    for s in range(0, len(logs0), bucket_sz):
        c0 = logs0[s : s + bucket_sz]
        c1 = logs1[s : s + bucket_sz]
        buckets.append(
            {
                "steps": f"{c0[0]['step']}-{c0[-1]['step']}",
                "avg_total_loss": round(
                    0.5 * (statistics.mean(x["total_loss"] for x in c0) + statistics.mean(x["total_loss"] for x in c1)),
                    4,
                ),
                "avg_lm_loss": round(
                    0.5
                    * (
                        statistics.mean(x["lm_letter_loss"] for x in c0)
                        + statistics.mean(x["lm_letter_loss"] for x in c1)
                    ),
                    4,
                ),
                "avg_aux_loss": round(
                    0.5
                    * (
                        statistics.mean(x["aux_head_loss"] for x in c0)
                        + statistics.mean(x["aux_head_loss"] for x in c1)
                    ),
                    4,
                ),
                "avg_step_sec": round(statistics.mean(x["step_sec"] for x in c0), 3),
            }
        )
    is_mp_logs = bool(logs0 and "gpu0_max_alloc_gb" in logs0[-1])
    summary = {
        "arm": arm,
        "model_parallel": is_mp_logs,
        "steps_completed": len(logs0),
        "steady_mean_sec_per_step": round(statistics.mean(steady), 3),
        "steady_median_sec_per_step": round(statistics.median(steady), 3),
        "gpu0_peak_allocated_gb": max(x["gpu0_max_alloc_gb"] for x in logs0) if is_mp_logs else max(x["max_alloc_gb"] for x in logs0),
        "gpu0_peak_reserved_gb": max(x["gpu0_max_reserved_gb"] for x in logs0) if is_mp_logs else max(x["reserved_gb"] for x in logs0),
        "gpu1_peak_allocated_gb": max(x["gpu1_max_alloc_gb"] for x in logs0) if is_mp_logs else max(x["max_alloc_gb"] for x in logs1),
        "gpu1_peak_reserved_gb": max(x["gpu1_max_reserved_gb"] for x in logs0) if is_mp_logs else max(x["reserved_gb"] for x in logs1),
        "loss_buckets": buckets,
    }
    Path(out_dir, "telemetry_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[Arm {arm} Summary] {json.dumps(summary, indent=2)}", flush=True)
    return summary


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.model_parallel or args.gpus <= 1:
        run_stage2_worker(0, 1, args)
        summarize_telemetry(args.output_dir, args.arm, 1)
        return

    mp.set_start_method("spawn", force=True)
    q0_to_1 = mp.Queue()
    q1_to_0 = mp.Queue()
    ctrl_q = mp.Queue()

    p0 = mp.Process(target=run_stage2_worker, args=(0, 2, args, q0_to_1, q1_to_0, ctrl_q))
    p1 = mp.Process(target=run_stage2_worker, args=(1, 2, args, q1_to_0, q0_to_1, ctrl_q))
    p0.start()
    p1.start()
    p0.join()
    p1.join()

    if p0.exitcode != 0 or p1.exitcode != 0:
        raise RuntimeError(f"Stage 2 worker failed: p0={p0.exitcode}, p1={p1.exitcode}")
    summarize_telemetry(args.output_dir, args.arm, 2)


if __name__ == "__main__":
    main()
