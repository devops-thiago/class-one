"""Shared prompt building and LM-head letter readout helpers for ClassOne Stage 2."""

from __future__ import annotations

import copy
import json
import os
import pathlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from classone.modeling.modeling_classone import ClassOneModel, _round_probability_distribution
from classone.schemas import ChoiceQuestion, ChoiceResult, NoulQuestion, Question, Result, ScoreQuestion
from classone.tokenizer import ClassOnePromptBuilder, PackedSequence, QuestionSpans

if TYPE_CHECKING:
    pass


GEMMA4_CODEBOOK_V1_PATH = pathlib.Path(__file__).resolve().parent / "codebooks" / "gemma4_e2b_codebook_v1.json"


def _extract_final_logit_softcapping(cfg: Any) -> float | None:
    """Extracts final_logit_softcapping from a config dict or PretrainedConfig object."""
    if cfg is None:
        return None
    if isinstance(cfg, Mapping):
        tc = cfg.get("text_config") or {}
        raw = tc.get("final_logit_softcapping", cfg.get("final_logit_softcapping"))
    else:
        tc = getattr(cfg, "text_config", None)
        raw = getattr(tc, "final_logit_softcapping", None) if tc is not None else None
        if raw is None:
            raw = getattr(cfg, "final_logit_softcapping", None)
    return float(raw) if raw is not None else None


def detect_model_family(obj: Any) -> str:
    """Detects model family ('gemma4', 'qwen', or 'default') from a model, config, path, or tokenizer."""
    if obj is None:
        return "default"

    if isinstance(obj, (str, pathlib.Path)):
        p = pathlib.Path(obj)
        cfg_file = p / "config.json"
        if cfg_file.exists():
            try:
                return detect_model_family(json.loads(cfg_file.read_text(encoding="utf-8")))
            except Exception:
                pass
        low = str(obj).lower()
        if "gemma4" in low or "gemma-4" in low:
            return "gemma4"
        if "qwen" in low:
            return "qwen"
        return "default"

    if isinstance(obj, Mapping):
        tc = obj.get("text_config") or {}
        parts = [
            str(obj.get("model_type", "")),
            str(tc.get("model_type", "")),
            " ".join(str(a) for a in (obj.get("architectures") or [])),
        ]
        joined = " ".join(parts).lower()
        if "gemma4" in joined or "gemma-4" in joined:
            return "gemma4"
        if "qwen" in joined:
            return "qwen"
        return "default"

    if hasattr(obj, "backbone") or hasattr(obj, "config"):
        cfg = getattr(obj, "config", None)
        if cfg is None and hasattr(obj, "backbone"):
            cfg = getattr(obj.backbone, "config", None)
        if cfg is not None and cfg is not obj:
            if isinstance(cfg, Mapping):
                res = detect_model_family(cfg)
            else:
                tc = getattr(cfg, "text_config", None)
                parts = [
                    str(getattr(cfg, "model_type", "") or ""),
                    str(getattr(tc, "model_type", "") or ""),
                    " ".join(str(a) for a in (getattr(cfg, "architectures", None) or [])),
                    str(getattr(cfg, "_name_or_path", "") or ""),
                ]
                joined = " ".join(parts).lower()
                if "gemma4" in joined or "gemma-4" in joined:
                    res = "gemma4"
                elif "qwen" in joined:
                    res = "qwen"
                else:
                    res = "default"
            if res != "default":
                return res

    # Tokenizer / string heuristics
    name_or_path = str(getattr(obj, "name_or_path", "") or "").lower()
    chat_template = str(getattr(obj, "chat_template", "") or "")
    if "gemma4" in name_or_path or "gemma-4" in name_or_path or "<|turn>" in chat_template:
        return "gemma4"
    if "qwen" in name_or_path or "<|im_start|>" in chat_template:
        return "qwen"
    return "default"


def build_gemma4_single_token_labels(
    tokenizer: PreTrainedTokenizerBase,
    max_k: int = 180,
    codebook_path: pathlib.Path | None = None,
) -> list[tuple[str, int]]:
    """Loads and validates the versioned Gemma 4 single-token round-trip codebook."""
    path = codebook_path or GEMMA4_CODEBOOK_V1_PATH
    yes_ids = set(tokenizer.encode("Yes", add_special_tokens=False))
    no_ids = set(tokenizer.encode("No", add_special_tokens=False))
    reserved_ids = yes_ids | no_ids

    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = payload.get("entries", [])
        validated: list[tuple[str, int]] = []
        all_match = len(entries) >= max_k
        for entry in entries[:max_k]:
            lbl = str(entry["label"])
            tid = int(entry["token_id"])
            ids = tokenizer.encode(lbl, add_special_tokens=False)
            dec = tokenizer.decode([tid], clean_up_tokenization_spaces=False) if len(ids) == 1 else ""
            if ids == [tid] and tid not in reserved_ids and lbl.isalnum() and dec == lbl:
                validated.append((lbl, tid))
            else:
                all_match = False
                break
        if all_match and len(validated) == max_k:
            return validated

    # Fallback for custom/mock Gemma-style tokenizers in unit tests: enforce strict round-trip + Yes/No disjointness.
    candidates = (
        [chr(code) for code in range(ord("A"), ord("Z") + 1)]
        + [chr(code) for code in range(ord("a"), ord("z") + 1)]
        + [str(i) for i in range(10)]
        + [chr(code) for code in range(0x0391, 0x03C9 + 1)]
        + [chr(code) for code in range(0x0410, 0x044F + 1)]
    )
    valid: list[tuple[str, int]] = []
    seen_ids = set(reserved_ids)
    for lbl in candidates:
        if not lbl.isalnum():
            continue
        ids = tokenizer.encode(lbl, add_special_tokens=False)
        if len(ids) != 1 or ids[0] in seen_ids:
            continue
        tid = int(ids[0])
        if tokenizer.decode([tid], clean_up_tokenization_spaces=False) != lbl:
            continue
        valid.append((lbl, tid))
        seen_ids.add(tid)
        if len(valid) >= max_k:
            return valid
    return valid


def strip_unused_multimodal_towers(backbone: Any, clear_cuda_cache: bool = True) -> list[str]:
    """Removes unused vision/audio towers from Qwen3.5 or Gemma4 backbones to reclaim VRAM."""
    if backbone is None:
        return []
    targets: list[Any] = [backbone]
    base_model = getattr(backbone, "base_model", None)
    inner = getattr(base_model, "model", None) if base_model is not None else None
    if inner is not None and inner is not backbone:
        targets.append(inner)

    stripped: list[str] = []
    for obj in targets:
        for attr in ("visual", "vision_tower", "audio_tower", "embed_vision", "embed_audio"):
            if getattr(obj, attr, None) is not None:
                try:
                    delattr(obj, attr)
                except Exception:
                    pass
                setattr(obj, attr, None)
                stripped.append(attr)

    if stripped and clear_cuda_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return stripped


@dataclass
class ChatPackedQuestion(PackedSequence):
    """Chat-formatted packed sequence carrying single-token LM-head label mappings."""

    label_strings: dict[str, list[str]] = field(default_factory=dict)
    label_token_ids: dict[str, list[int]] = field(default_factory=dict)


def build_single_token_labels(tokenizer: PreTrainedTokenizerBase, max_k: int = 180) -> list[tuple[str, int]]:
    """Builds a deterministic codebook of single-token labels ordered A..Z first."""
    candidates = (
        [chr(ord("A") + i) for i in range(26)]
        + [chr(ord("a") + i) for i in range(26)]
        + [str(i) for i in range(100)]
        + [chr(code) for code in range(0x0391, 0x03C9)]
        + [chr(code) for code in range(0x0410, 0x044F)]
    )
    valid: list[tuple[str, int]] = []
    seen_ids: set[int] = set()
    for lbl in candidates:
        ids = tokenizer.encode(lbl, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in seen_ids:
            valid.append((lbl, ids[0]))
            seen_ids.add(ids[0])
            if len(valid) >= max_k:
                return valid

    for tid in range(100, 12000):
        if tid in seen_ids:
            continue
        s = tokenizer.decode([tid]).strip()
        if 2 <= len(s) <= 4 and s.isalnum() and tokenizer.encode(s, add_special_tokens=False) == [tid]:
            valid.append((s, tid))
            seen_ids.add(tid)
            if len(valid) >= max_k:
                break
    return valid


def _clean_choice_options(q: ChoiceQuestion) -> list[tuple[str, str]]:
    cleaned: list[tuple[str, str]] = []
    for opt_key, opt_desc in q.criteria.items():
        if opt_desc is None:
            clean_desc = ""
        elif isinstance(opt_desc, str):
            clean_desc = opt_desc.strip()
        elif isinstance(opt_desc, dict):
            raw_what = opt_desc.get("what")
            clean_desc = (str(raw_what) if raw_what is not None else str(opt_desc)).strip()
        else:
            clean_desc = ClassOnePromptBuilder._serialize_text_field(opt_desc)
        if not clean_desc or clean_desc.lower() == "none":
            k_low = str(opt_key).strip().lower()
            if k_low in ("yes", "true"):
                clean_desc = "Yes — affirmative, the condition holds."
            elif k_low in ("no", "false"):
                clean_desc = "No — negative, the condition does not hold."
            else:
                clean_desc = str(opt_key)
        cleaned.append((str(opt_key), clean_desc))
    return cleaned


class ChatDecisionPromptBuilder:
    """Formats ClassOne decision requests into native chat prompts with tracked spans."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_k: int = 180,
        model_family: str | None = None,
        codebook_path: pathlib.Path | str | None = None,
    ):
        self.tokenizer = tokenizer
        self.model_family = model_family or detect_model_family(tokenizer)
        if self.model_family == "gemma4":
            cb_path = pathlib.Path(codebook_path) if codebook_path is not None else None
            self.codebook = build_gemma4_single_token_labels(tokenizer, max_k=max_k, codebook_path=cb_path)
            yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
            no_ids = tokenizer.encode("No", add_special_tokens=False)
            if len(yes_ids) != 1 or len(no_ids) != 1:
                raise ValueError(f"Gemma4 Yes/No must each encode to a single token, got Yes={yes_ids}, No={no_ids}")
            self.yes_no_ids = [int(yes_ids[0]), int(no_ids[0])]
        else:
            self.codebook = build_single_token_labels(tokenizer, max_k=max_k)
            yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
            no_ids = tokenizer.encode("No", add_special_tokens=False)
            self.yes_no_ids = [yes_ids[0], no_ids[0]]

        # Cache template assistant tail (e.g. "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        sentinel = "__CLASSONE_SENTINEL__"
        if self.model_family == "gemma4":
            try:
                rendered = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": sentinel}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                rendered = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": sentinel}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            rendered = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": sentinel}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        prefix, tail = rendered.split(sentinel, 1)
        self.user_prefix = prefix
        self.assistant_tail = tail

    @staticmethod
    def serialize_state(state: Any) -> str:
        if isinstance(state, str):
            return state.strip()
        return json.dumps(state, indent=2, ensure_ascii=False)

    def format_shared_prefix(
        self,
        state: Any,
        sibling_questions: Mapping[str, Question] | None = None,
    ) -> str:
        """Builds the user-turn prefix up to (but not including) a specific question task."""
        state_str = self.serialize_state(state)
        parts = [f"{self.user_prefix}Context:\n{state_str}\n\n"]
        if sibling_questions and len(sibling_questions) > 1:
            overview = "\n".join(
                f"- [{qid}]: {ClassOnePromptBuilder._serialize_text_field(getattr(q, 'instructions', '')).strip()[:240]}"
                for qid, q in sibling_questions.items()
            )
            parts.append(f"All Candidate Decisions in Request:\n{overview}\n\n")
        return "".join(parts)

    def pack_question(
        self,
        state: Any,
        q_id: str,
        question: Question,
        sibling_questions: Mapping[str, Question] | None = None,
    ) -> ChatPackedQuestion:
        """Packs a single question into the native chat template with exact option and query spans."""
        shared_prefix = self.format_shared_prefix(state, sibling_questions=sibling_questions)
        current_ids = self.tokenizer.encode(shared_prefix, add_special_tokens=False)
        text_parts = [shared_prefix]

        raw_inst = getattr(question, "instructions", "")
        if isinstance(raw_inst, dict):
            inst_text = json.dumps(raw_inst).strip()
        else:
            inst_text = ClassOnePromptBuilder._serialize_text_field(raw_inst).strip()

        if isinstance(question, NoulQuestion):
            body = f"Task: {inst_text}\nRespond with Yes or No.{self.assistant_tail}"
            body_ids = self.tokenizer.encode(body, add_special_tokens=False)
            current_ids.extend(body_ids)
            text_parts.append(body)
            query_idx = len(current_ids) - 1
            span = QuestionSpans(
                question_id=q_id,
                question_type="noul",
                query_token_idx=query_idx,
                option_keys=["yes", "no"],
            )
            inp_t = torch.tensor([current_ids], dtype=torch.long)
            return ChatPackedQuestion(
                text="".join(text_parts),
                input_ids=inp_t,
                attention_mask=torch.ones_like(inp_t),
                questions={q_id: span},
                label_strings={q_id: ["Yes", "No"]},
                label_token_ids={q_id: list(self.yes_no_ids)},
            )

        if isinstance(question, ChoiceQuestion):
            cleaned_options = _clean_choice_options(question)
            k_len = len(cleaned_options)
            assert k_len <= len(self.codebook), f"Option count {k_len} exceeds codebook {len(self.codebook)}"
            cb = self.codebook[:k_len]

            header = f"Task: {inst_text}\n\nOptions:\n"
            current_ids.extend(self.tokenizer.encode(header, add_special_tokens=False))
            text_parts.append(header)

            opt_keys: list[str] = []
            opt_indices: list[int] = []
            opt_spans: list[tuple[int, int]] = []
            lbl_strs: list[str] = []
            lbl_ids: list[int] = []

            for i, (opt_key, opt_desc) in enumerate(cleaned_options):
                lbl_s, lbl_id = cb[i]
                opt_keys.append(opt_key)
                lbl_strs.append(lbl_s)
                lbl_ids.append(lbl_id)
                line = f"{lbl_s}. {opt_desc}\n"
                line_ids = self.tokenizer.encode(line, add_special_tokens=False)
                start_off = len(current_ids)
                current_ids.extend(line_ids)
                end_off = len(current_ids)
                opt_spans.append((start_off, end_off))
                opt_indices.append(end_off - 1)
                text_parts.append(line)

            tail = f"\nRespond with ONLY the exact option label ({cb[0][0]}..{cb[-1][0]}).{self.assistant_tail}"
            current_ids.extend(self.tokenizer.encode(tail, add_special_tokens=False))
            text_parts.append(tail)

            query_idx = len(current_ids) - 1
            span = QuestionSpans(
                question_id=q_id,
                question_type="choice",
                query_token_idx=query_idx,
                option_keys=opt_keys,
                option_token_indices=opt_indices,
                option_token_spans=opt_spans,
            )
            inp_t = torch.tensor([current_ids], dtype=torch.long)
            return ChatPackedQuestion(
                text="".join(text_parts),
                input_ids=inp_t,
                attention_mask=torch.ones_like(inp_t),
                questions={q_id: span},
                label_strings={q_id: lbl_strs},
                label_token_ids={q_id: lbl_ids},
            )

        if isinstance(question, ScoreQuestion):
            levels = [str(lv) for lv in question.criteria]
            k_len = len(levels)
            cb = self.codebook[:k_len]
            header = f"Task: {inst_text}\n\nRubric Levels:\n"
            current_ids.extend(self.tokenizer.encode(header, add_special_tokens=False))
            text_parts.append(header)

            lvl_keys: list[str] = []
            lvl_indices: list[int] = []
            lvl_spans: list[tuple[int, int]] = []
            lbl_strs = []
            lbl_ids = []

            for i, lvl_desc in enumerate(levels):
                lbl_s, lbl_id = cb[i]
                lvl_keys.append(str(i + 1))
                lbl_strs.append(lbl_s)
                lbl_ids.append(lbl_id)
                line = f"{lbl_s}. {lvl_desc}\n"
                line_ids = self.tokenizer.encode(line, add_special_tokens=False)
                start_off = len(current_ids)
                current_ids.extend(line_ids)
                end_off = len(current_ids)
                lvl_spans.append((start_off, end_off))
                lvl_indices.append(end_off - 1)
                text_parts.append(line)

            tail = f"\nRespond with ONLY the exact level label ({cb[0][0]}..{cb[-1][0]}).{self.assistant_tail}"
            current_ids.extend(self.tokenizer.encode(tail, add_special_tokens=False))
            text_parts.append(tail)

            query_idx = len(current_ids) - 1
            span = QuestionSpans(
                question_id=q_id,
                question_type="score",
                query_token_idx=query_idx,
                option_keys=lvl_keys,
                option_token_indices=lvl_indices,
                option_token_spans=lvl_spans,
            )
            inp_t = torch.tensor([current_ids], dtype=torch.long)
            return ChatPackedQuestion(
                text="".join(text_parts),
                input_ids=inp_t,
                attention_mask=torch.ones_like(inp_t),
                questions={q_id: span},
                label_strings={q_id: lbl_strs},
                label_token_ids={q_id: lbl_ids},
            )

        raise ValueError(f"Unsupported question type: {type(question)}")


class CodebookLMHead(nn.Module):
    """Frozen lm_head/embed_tokens row slice covering single-token codebook and Yes/No label IDs."""

    def __init__(
        self,
        weight_slice: torch.Tensor,
        token_ids: list[int],
        final_logit_softcapping: float | None = None,
    ):
        super().__init__()
        if weight_slice.ndim != 2 or weight_slice.shape[0] != len(token_ids):
            raise ValueError(
                f"Expected weight_slice shape ({len(token_ids)}, hidden_size), got {tuple(weight_slice.shape)}"
            )
        self.register_buffer("weight", weight_slice, persistent=True)
        self.token_ids = [int(t) for t in token_ids]
        self.token_to_row = {int(t): idx for idx, t in enumerate(self.token_ids)}
        self.final_logit_softcapping = float(final_logit_softcapping) if final_logit_softcapping is not None else None

    def select_rows(self, label_token_ids: list[int]) -> torch.Tensor:
        """Selects rows matching requested vocabulary token IDs."""
        rows = []
        for tid in label_token_ids:
            tid_int = int(tid)
            if tid_int not in self.token_to_row:
                raise KeyError(f"Token ID {tid_int} is not in CodebookLMHead slice ({len(self.token_ids)} tokens).")
            rows.append(self.token_to_row[tid_int])
        row_idx = torch.tensor(rows, device=self.weight.device, dtype=torch.long)
        return self.weight.index_select(0, row_idx)

    def forward(self, h_query: torch.Tensor, label_token_ids: list[int] | None = None) -> torch.Tensor:
        """Projects query hidden vector(s) onto sliced vocabulary rows with optional softcapping."""
        w = self.weight if label_token_ids is None else self.select_rows(label_token_ids)
        logits = F.linear(h_query.to(w.dtype), w)
        if self.final_logit_softcapping is not None and self.final_logit_softcapping > 0.0:
            logits = torch.tanh(logits / self.final_logit_softcapping) * self.final_logit_softcapping
        return logits


def _resolve_local_model_dir(model_path: str) -> pathlib.Path:
    p = pathlib.Path(model_path)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return pathlib.Path(snapshot_download(model_path, local_files_only=True))


def load_codebook_lm_head(
    model_path: str,
    chat_builder: ChatDecisionPromptBuilder,
    device: str | torch.device,
    dtype: torch.dtype = torch.bfloat16,
    include_yes_no: bool = True,
    allow_tied: bool = False,
    model_family: str | None = None,
) -> CodebookLMHead | None:
    """Loads a frozen codebook row slice from untied lm_head or Gemma 4 tied embed_tokens."""
    from safetensors import safe_open

    model_dir = _resolve_local_model_dir(model_path)
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    family = model_family or detect_model_family(cfg)
    tie_top = cfg.get("tie_word_embeddings")
    tie_tc = (cfg.get("text_config") or {}).get("tie_word_embeddings")
    is_tied = tie_top is not False and tie_tc is not False
    if is_tied and family != "gemma4" and not allow_tied:
        return None

    softcap = _extract_final_logit_softcapping(cfg)
    key_candidates = (
        [
            "model.language_model.embed_tokens.weight",
            "language_model.embed_tokens.weight",
            "model.embed_tokens.weight",
            "embed_tokens.weight",
            "lm_head.weight",
        ]
        if is_tied
        else ["lm_head.weight"]
    )

    idx_path = model_dir / "model.safetensors.index.json"
    target_key: str | None = None
    shard_path: pathlib.Path | None = None
    if idx_path.exists():
        weight_map = json.loads(idx_path.read_text(encoding="utf-8")).get("weight_map", {})
        for cand in key_candidates:
            if cand in weight_map:
                target_key = cand
                shard_path = model_dir / weight_map[cand]
                break
    elif (model_dir / "model.safetensors").exists():
        shard_path = model_dir / "model.safetensors"

    if shard_path is None or not shard_path.exists():
        return None

    token_ids: list[int] = []
    seen: set[int] = set()
    for _, tid in chat_builder.codebook:
        tid_int = int(tid)
        if tid_int not in seen:
            token_ids.append(tid_int)
            seen.add(tid_int)
    if include_yes_no:
        for tid in chat_builder.yes_no_ids:
            tid_int = int(tid)
            if tid_int not in seen:
                token_ids.append(tid_int)
                seen.add(tid_int)

    with safe_open(str(shard_path), framework="pt", device="cpu") as f:
        avail = set(f.keys())
        if target_key is None:
            for cand in key_candidates:
                if cand in avail:
                    target_key = cand
                    break
        if target_key is None or target_key not in avail:
            return None
        full_w = f.get_tensor(target_key)
        idx_t = torch.tensor(token_ids, dtype=torch.long)
        slice_w = full_w.index_select(0, idx_t).to(dtype=dtype).contiguous().to(device)
        del full_w

    return CodebookLMHead(slice_w, token_ids, final_logit_softcapping=softcap)


def compute_lm_letter_logits(
    h_query: torch.Tensor,
    embed_weight: torch.Tensor | CodebookLMHead,
    label_token_ids: list[int],
    scale: float = 1.0,
    final_logit_softcapping: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scores candidate single-token option labels through a codebook slice or full vocab embedding matrix."""
    if hasattr(embed_weight, "select_rows"):
        w_cand = embed_weight.select_rows(label_token_ids)
    else:
        cand_ids = torch.tensor(label_token_ids, device=embed_weight.device, dtype=torch.long)
        w_cand = embed_weight.index_select(0, cand_ids)
    if w_cand.device != h_query.device:
        w_cand = w_cand.to(h_query.device)
    softcap = (
        float(final_logit_softcapping)
        if final_logit_softcapping is not None
        else getattr(embed_weight, "final_logit_softcapping", None)
    )
    # Apply optional softcapping in weight dtype (matching HF Gemma4 forward), then cast to FP32 before scaling
    logits = F.linear(h_query.to(w_cand.dtype).unsqueeze(0), w_cand).squeeze(0)
    if softcap is not None and softcap > 0.0:
        logits = torch.tanh(logits / softcap) * softcap
    logits = logits.float() * scale
    probs = F.softmax(logits, dim=-1)
    return probs, logits


def copy_pkv(base_pkv: Any) -> Any:
    """Clones hybrid attention + linear conv/recurrent cache states for branched inference."""
    return copy.deepcopy(base_pkv)


def _resolve_component_adapter_dir(snap_path: pathlib.Path, name: str, weight: float) -> pathlib.Path:
    """Finds the PEFT LoRA adapter directory for a merged snapshot component."""
    candidates = [
        snap_path / f"component_{weight:.2f}_{name}",
        snap_path / f"component_{weight}_{name}",
        snap_path / name,
    ]
    for cand in candidates:
        if (cand / "adapter_config.json").exists():
            return cand
    for child in sorted(snap_path.iterdir()):
        if child.is_dir() and name in child.name and (child / "adapter_config.json").exists():
            return child
    raise FileNotFoundError(f"Could not locate adapter_config.json for component '{name}' inside {snap_path}")


def _apply_and_merge_lora(model: ClassOneModel, adapter_dir: pathlib.Path, scale: float = 1.0) -> None:
    """Applies a PEFT LoRA adapter onto model.backbone with optional weight scaling and merges it."""
    from peft import PeftModel

    model.backbone = PeftModel.from_pretrained(model.backbone, str(adapter_dir))
    if scale != 1.0:
        for mod in model.backbone.modules():
            if hasattr(mod, "scaling") and isinstance(mod.scaling, dict):
                for key in list(mod.scaling.keys()):
                    mod.scaling[key] = float(mod.scaling[key]) * float(scale)
    model.backbone = model.backbone.merge_and_unload()


def load_stage2_snapshot(
    base_model_path: str,
    snapshot_dir: str,
    device: str = "cuda",
    quantization: str | None = None,
) -> tuple[ClassOneModel, PreTrainedTokenizerBase, ChatDecisionPromptBuilder]:
    """Loads a Stage 2 merged or adapter snapshot onto the raw base model for inference."""
    snap_path = pathlib.Path(snapshot_dir)
    local_only = os.path.isdir(base_model_path) or os.environ.get("HF_HUB_OFFLINE") == "1"
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        local_files_only=local_only,
    )
    chat_builder = ChatDecisionPromptBuilder(copy.deepcopy(tokenizer))

    model_dtype = torch.bfloat16 if device != "cpu" and not quantization else None
    model = ClassOneModel.from_backbone(
        base_model_path,
        tokenizer=None,
        device=device,
        quantization=quantization,
        torch_dtype=model_dtype,
    )

    meta_path = snap_path / "snapshot_metadata.json"
    merged_components: list[dict[str, Any]] = []
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        merged_components = meta.get("merge_recipe", {}).get("components", []) or []

    if merged_components:
        for comp in merged_components:
            weight = float(comp.get("weight", 1.0))
            if weight == 0.0:
                continue
            comp_dir = _resolve_component_adapter_dir(snap_path, str(comp["name"]), weight)
            _apply_and_merge_lora(model, comp_dir, scale=weight)
    else:
        lora_dir = snap_path / "lora_backbone"
        if lora_dir.exists():
            _apply_and_merge_lora(model, lora_dir, scale=1.0)

    heads_path = snap_path / "classone_heads.pt"
    if heads_path.exists():
        heads = torch.load(heads_path, map_location=model.device, weights_only=True)
        model.noul_head.load_state_dict(heads["noul_head"], strict=False)
        model.choice_head.load_state_dict(heads["choice_head"], strict=False)
        model.score_head.load_state_dict(heads["score_head"], strict=False)

    if getattr(model, "lm_readout_head", None) is None and not quantization:
        cb_dtype = model_dtype if model_dtype is not None else model._backbone_float_dtype()
        codebook_head = load_codebook_lm_head(
            base_model_path,
            chat_builder,
            device=model.output_device,
            dtype=cb_dtype,
            include_yes_no=True,
        )
        if codebook_head is not None:
            model.lm_readout_head = codebook_head

    strip_unused_multimodal_towers(model.backbone, clear_cuda_cache=torch.cuda.is_available())

    model.eval()
    return model, tokenizer, chat_builder


def _pack_with_truncation(
    chat_builder: ChatDecisionPromptBuilder,
    state: Any,
    q_id: str,
    question: Question,
    max_tokens: int,
    truncate_chars: int,
) -> ChatPackedQuestion:
    """Packs a Stage 2 chat question, truncating serialized state if it exceeds max_tokens."""
    packed = chat_builder.pack_question(state=state, q_id=q_id, question=question)
    if packed.input_ids.shape[1] > max_tokens:
        s_str = chat_builder.serialize_state(state)
        packed = chat_builder.pack_question(state=s_str[:truncate_chars], q_id=q_id, question=question)
    return packed


def _eval_stage2_choice_single_pass(
    model: ClassOneModel,
    chat_builder: ChatDecisionPromptBuilder,
    embed_w: torch.Tensor | CodebookLMHead,
    state: Any,
    q_id: str,
    question: ChoiceQuestion,
    max_tokens: int,
) -> tuple[str, torch.Tensor, list[str], int]:
    """Runs a single Stage 2 forward pass for a ChoiceQuestion, returning (best_key, logits, opt_keys, tokens)."""
    packed = _pack_with_truncation(
        chat_builder,
        state=state,
        q_id=q_id,
        question=question,
        max_tokens=max_tokens,
        truncate_chars=max_tokens * 2,
    )
    seq_tokens = int(packed.input_ids.shape[1])
    in_dev = model.input_device
    with torch.no_grad():
        hs = model.extract_hidden_states(
            input_ids=packed.input_ids.to(in_dev),
            attention_mask=packed.attention_mask.to(in_dev),
        )
        span = packed.questions[q_id]
        h_q = hs[0][span.query_token_idx]
        _, logits_lm = compute_lm_letter_logits(
            h_q,
            embed_w,
            packed.label_token_ids[q_id],
            scale=1.0,
            final_logit_softcapping=(
                None
                if getattr(embed_w, "final_logit_softcapping", None) is not None
                else _extract_final_logit_softcapping(getattr(model.backbone, "config", None))
            ),
        )
    logits_cpu = logits_lm.cpu()
    best_idx = int(torch.argmax(logits_cpu).item())
    return span.option_keys[best_idx], logits_cpu, span.option_keys, seq_tokens


def evaluate_stage2_request(
    model: ClassOneModel,
    chat_builder: ChatDecisionPromptBuilder,
    state: Any,
    questions: Mapping[str, Question],
    m2_k_threshold: int = 64,
    m2_top_k: int = 16,
    max_tokens: int = 4096,
) -> tuple[dict[str, Result], int]:
    """Evaluates a DecisionRequest using Stage 2 LM-head letter readout and M2 shortlist routing."""
    model.eval()
    if getattr(model, "lm_readout_head", None) is not None:
        embed_w = model.lm_readout_head
    elif hasattr(model, "embed"):
        embed_w = model.embed.weight
    else:
        embed_w = model.backbone.get_input_embeddings().weight
    head_dtype = next(model.noul_head.parameters()).dtype
    in_dev = model.input_device
    answers: dict[str, Result] = {}
    total_input_tokens = 0

    for q_id, question in questions.items():
        if isinstance(question, ChoiceQuestion):
            best_k, z_lm, opt_keys, pass1_tokens = _eval_stage2_choice_single_pass(
                model,
                chat_builder,
                embed_w,
                state=state,
                q_id=q_id,
                question=question,
                max_tokens=max_tokens,
            )
            total_input_tokens += pass1_tokens

            if len(opt_keys) >= m2_k_threshold:
                top_indices = torch.topk(z_lm, min(m2_top_k, len(opt_keys))).indices.tolist()
                sub_crit = {opt_keys[i]: question.criteria[opt_keys[i]] for i in top_indices}
                sub_q = ChoiceQuestion(instructions=question.instructions, criteria=sub_crit)
                best_k, z_m2, sub_keys, pass2_tokens = _eval_stage2_choice_single_pass(
                    model,
                    chat_builder,
                    embed_w,
                    state=state,
                    q_id=q_id,
                    question=sub_q,
                    max_tokens=max_tokens,
                )
                total_input_tokens += pass2_tokens
                p1 = torch.softmax(z_lm, dim=-1).tolist()
                p2 = torch.softmax(z_m2, dim=-1).tolist()
                probs = [x * 1e-3 for x in p1]
                for ks, ps in zip(sub_keys, p2):
                    probs[opt_keys.index(ks)] = ps
                stot = sum(probs) or 1.0
                probs = [x / stot for x in probs]
            else:
                probs = torch.softmax(z_lm, dim=-1).tolist()

            p_dict = _round_probability_distribution(opt_keys, probs)
            conf = (
                max(0.0, min(1.0, (len(opt_keys) * max(probs) - 1.0) / max(1, len(opt_keys) - 1)))
                if len(opt_keys) > 1
                else 1.0
            )
            sp = sorted(probs, reverse=True)
            margin = float(sp[0] - sp[1]) if len(sp) > 1 else 1.0
            answers[q_id] = ChoiceResult(
                choice=best_k,
                probabilities=p_dict,
                confidence=round(conf, 4),
                is_decisive=conf >= 0.15,
                margin=round(margin, 4),
            )
        else:
            packed = _pack_with_truncation(
                chat_builder,
                state=state,
                q_id=q_id,
                question=question,
                max_tokens=max_tokens,
                truncate_chars=min(8000, max_tokens * 2),
            )
            total_input_tokens += int(packed.input_ids.shape[1])
            with torch.no_grad():
                hs = model.extract_hidden_states(
                    packed.input_ids.to(in_dev),
                    packed.attention_mask.to(in_dev),
                )
                answers[q_id] = model._score_question_span(
                    hs[0].to(head_dtype),
                    packed.questions[q_id],
                )

    return answers, total_input_tokens
