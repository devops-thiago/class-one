"""Prompt formatting, special tokens, and token span tracking for ClassOne.

Serializes state and questions into a single structured sequence, tracking the
exact token index of decision delimiter tokens (<|noul_end|>, <|choice_end|>, etc.)
for single-pass parallel pooling.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from classone.schemas import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion

# Reserved delimiter tokens for ClassOne decision architecture
SPECIAL_TOKENS = [
    "<|state_start|>",
    "<|state_end|>",
    "<|noul_start|>",
    "<|noul_end|>",
    "<|choice_start|>",
    "<|choice_end|>",
    "<|opt_start|>",
    "<|opt_end|>",
    "<|score_start|>",
    "<|score_end|>",
    "<|level_start|>",
    "<|level_end|>",
    "<|pause|>",
]


@dataclass
class QuestionSpans:
    """Token index locations for a single question within the input token sequence."""

    question_id: str
    question_type: str
    query_token_idx: int
    option_keys: list[str] = field(default_factory=list)
    option_token_indices: list[int] = field(default_factory=list)
    option_token_spans: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class PackedSequence:
    """Packed prompt tokens and question span indices ready for model forward pass."""

    text: str
    input_ids: torch.Tensor  # shape: [1, seq_len]
    attention_mask: torch.Tensor  # shape: [1, seq_len]
    questions: dict[str, QuestionSpans]


class ClassOnePromptBuilder:
    """Builds structured ClassOne decision prompts and maps delimiter token spans."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, default_pause_tokens: int = 0):
        self.tokenizer = tokenizer
        self.default_pause_tokens = default_pause_tokens
        # Check for Gemma in-vocab reserved tokens (<unused0>..<unusedN>)
        # Gemma models MUST always use in-vocab mapping to protect the 42 Per-Layer Embeddings (PLE)
        # and prevent out-of-bounds embedding gather asserts.
        has_unused = (
            hasattr(tokenizer, "convert_tokens_to_ids")
            and tokenizer.convert_tokens_to_ids("<unused0>") is not None
            and tokenizer.convert_tokens_to_ids("<unused0>") != getattr(tokenizer, "unk_token_id", -1)
        )

        if has_unused:
            # Map delimiters to native in-vocab reserved tokens (<unused0>..<unused12>)
            # This avoids resizing the 42-layer Per-Layer Embeddings (PLE) in Gemma 4, saving 5.25 GB VRAM
            self.delimiter_map = {special: f"<unused{i}>" for i, special in enumerate(SPECIAL_TOKENS)}
            self.special_token_ids = {
                special: self.tokenizer.convert_tokens_to_ids(f"<unused{i}>")
                for i, special in enumerate(SPECIAL_TOKENS)
            }
        else:
            self.delimiter_map = {tok: tok for tok in SPECIAL_TOKENS}
            self.tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
            self.special_token_ids = {tok: self.tokenizer.convert_tokens_to_ids(tok) for tok in SPECIAL_TOKENS}

    def serialize_state(self, state: str | dict[str, Any] | list[Any]) -> str:
        """Serializes state into text."""
        if isinstance(state, str):
            return state.strip()
        return json.dumps(state, indent=2, ensure_ascii=False)

    @staticmethod
    def _serialize_text_field(value: Any) -> str:
        """Deterministically serialize str/dict/list/None prompt fields to text."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    def pack(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, Question],
        num_pause_tokens: int | None = None,
        options_in_prefix: bool | str = False,
        preamble_questions: Mapping[str, Question] | None = None,
    ) -> PackedSequence:
        """Packs state and questions into a single sequence with exact token positions.

        To guarantee exact token index tracking without tokenizer ambiguity, tokens
        are constructed segment-by-segment.
        """
        if num_pause_tokens is None:
            num_pause_tokens = self.default_pause_tokens

        d_state_start = self.delimiter_map["<|state_start|>"]
        d_state_end = self.delimiter_map["<|state_end|>"]
        d_noul_start = self.delimiter_map["<|noul_start|>"]
        d_noul_end = self.delimiter_map["<|noul_end|>"]
        d_choice_start = self.delimiter_map["<|choice_start|>"]
        d_choice_end = self.delimiter_map["<|choice_end|>"]
        d_opt_start = self.delimiter_map["<|opt_start|>"]
        d_opt_end = self.delimiter_map["<|opt_end|>"]
        d_score_start = self.delimiter_map["<|score_start|>"]
        d_score_end = self.delimiter_map["<|score_end|>"]
        d_level_start = self.delimiter_map["<|level_start|>"]
        d_level_end = self.delimiter_map["<|level_end|>"]
        d_pause = self.delimiter_map["<|pause|>"]

        state_str = self.serialize_state(state)
        state_prefix = f"{d_state_start}\n{state_str}\n{d_state_end}\n"
        prefix_ids = self.tokenizer.encode(
            state_prefix,
            add_special_tokens=True,
            return_tensors=None,
        )

        question_spans_meta: dict[str, QuestionSpans] = {}

        # For long context documents (>600 tokens), condition causal attention with a question preamble
        text_parts: list[str] = []
        current_ids: list[int] = []

        target_preamble_qs = preamble_questions if preamble_questions is not None else questions
        if len(prefix_ids) > 600 and target_preamble_qs:
            instructions_summary = "\n".join(
                f"- [{qid}]: {self._serialize_text_field(getattr(q, 'instructions', None)).strip()}"
                for qid, q in target_preamble_qs.items()
            )
            preamble_text = f"{d_state_start}\nTarget Objectives:\n{instructions_summary}\n{d_state_end}\n"
            preamble_ids = self.tokenizer.encode(preamble_text, add_special_tokens=True, return_tensors=None)
            text_parts.append(preamble_text)
            current_ids.extend(preamble_ids)

            state_only_ids = self.tokenizer.encode(state_prefix, add_special_tokens=False, return_tensors=None)
            text_parts.append(state_prefix)
            current_ids.extend(state_only_ids)
        else:
            text_parts.append(state_prefix)
            current_ids.extend(prefix_ids)

        # Inject latent pause tokens before questions to give transformer layers sequence depth
        if num_pause_tokens > 0:
            pause_text = (f"{d_pause} " * num_pause_tokens).strip() + "\n"
            pause_ids = self.tokenizer.encode(pause_text, add_special_tokens=False, return_tensors=None)
            current_ids.extend(pause_ids)
            text_parts.append(pause_text)

        for q_id, q in questions.items():
            inst_text = self._serialize_text_field(q.instructions)
            if isinstance(q, NoulQuestion):
                q_text = f"{d_noul_start}\nQuestion ID: {q_id}\nInstruction: {inst_text}\n{d_noul_end}\n"
                seg_ids = self.tokenizer.encode(q_text, add_special_tokens=False, return_tensors=None)
                start_offset = len(current_ids)
                current_ids.extend(seg_ids)
                text_parts.append(q_text)

                # Locate noul_end token position in this segment
                end_tok_id = self.special_token_ids["<|noul_end|>"]
                end_indices = [idx for idx, tid in enumerate(seg_ids) if tid == end_tok_id]
                if not end_indices:
                    query_idx = len(current_ids) - 1
                else:
                    query_idx = start_offset + end_indices[-1]

                question_spans_meta[q_id] = QuestionSpans(
                    question_id=q_id,
                    question_type="noul",
                    query_token_idx=query_idx,
                )

            elif isinstance(q, ChoiceQuestion):
                cleaned_options: list[tuple[str, str]] = []
                for opt_key, opt_desc in q.criteria.items():
                    if opt_desc is None:
                        clean_desc = ""
                    elif isinstance(opt_desc, str):
                        clean_desc = opt_desc.strip()
                    else:
                        clean_desc = self._serialize_text_field(opt_desc)
                    if not clean_desc or (isinstance(opt_desc, str) and clean_desc.lower() == "none"):
                        k_low = str(opt_key).strip().lower()
                        if k_low in ("yes", "true"):
                            clean_desc = "Yes — affirmative, the condition holds."
                        elif k_low in ("no", "false"):
                            clean_desc = "No — negative, the condition does not hold."
                        else:
                            clean_desc = str(opt_key)
                    cleaned_options.append((opt_key, clean_desc))

                # Choice preamble: list candidate options in prefix only when options_in_prefix is enabled
                # and the options are substantive (never for repeated boilerplate yes/no or true/false options)
                use_opts_in_prefix = bool(options_in_prefix) and has_substantive_options(cleaned_options)
                if use_opts_in_prefix and cleaned_options:
                    options_listing = "\n".join(f"- {k}: {d}" for k, d in cleaned_options)
                    choice_header = (
                        f"{d_choice_start}\nQuestion ID: {q_id}\nInstruction: {inst_text}\n"
                        f"Candidate Options:\n{options_listing}\n"
                    )
                else:
                    choice_header = f"{d_choice_start}\nQuestion ID: {q_id}\nInstruction: {inst_text}\n"
                header_ids = self.tokenizer.encode(choice_header, add_special_tokens=False, return_tensors=None)
                current_ids.extend(header_ids)
                text_parts.append(choice_header)

                opt_keys: list[str] = []
                opt_indices: list[int] = []
                opt_spans: list[tuple[int, int]] = []

                for opt_key, clean_desc in cleaned_options:
                    opt_keys.append(opt_key)
                    opt_text = f"{d_opt_start}\nKey: {opt_key}\nCriteria: {clean_desc}\n{d_opt_end}\n"
                    opt_ids = self.tokenizer.encode(opt_text, add_special_tokens=False, return_tensors=None)
                    start_offset = len(current_ids)
                    current_ids.extend(opt_ids)
                    end_offset = len(current_ids)
                    opt_spans.append((start_offset, end_offset))
                    text_parts.append(opt_text)

                    end_tok_id = self.special_token_ids["<|opt_end|>"]
                    end_idx = [idx for idx, tid in enumerate(opt_ids) if tid == end_tok_id]
                    if not end_idx:
                        opt_indices.append(len(current_ids) - 1)
                    else:
                        opt_indices.append(start_offset + end_idx[-1])

                choice_footer = f"{d_choice_end}\n"
                footer_ids = self.tokenizer.encode(choice_footer, add_special_tokens=False, return_tensors=None)
                start_offset = len(current_ids)
                current_ids.extend(footer_ids)
                text_parts.append(choice_footer)

                end_tok_id = self.special_token_ids["<|choice_end|>"]
                end_idx = [idx for idx, tid in enumerate(footer_ids) if tid == end_tok_id]
                query_idx = start_offset + end_idx[-1] if end_idx else len(current_ids) - 1

                question_spans_meta[q_id] = QuestionSpans(
                    question_id=q_id,
                    question_type="choice",
                    query_token_idx=query_idx,
                    option_keys=opt_keys,
                    option_token_indices=opt_indices,
                    option_token_spans=opt_spans,
                )

            elif isinstance(q, ScoreQuestion):
                score_header = f"{d_score_start}\nQuestion ID: {q_id}\nInstruction: {inst_text}\n"
                header_ids = self.tokenizer.encode(score_header, add_special_tokens=False, return_tensors=None)
                current_ids.extend(header_ids)
                text_parts.append(score_header)

                level_keys: list[str] = []
                level_indices: list[int] = []
                level_spans: list[tuple[int, int]] = []

                for idx, level_desc in enumerate(q.criteria):
                    level_key = str(idx + 1)
                    level_keys.append(level_key)
                    level_text = f"{d_level_start}\nLevel: {level_key}\nCriteria: {level_desc}\n{d_level_end}\n"
                    level_ids = self.tokenizer.encode(level_text, add_special_tokens=False, return_tensors=None)
                    start_offset = len(current_ids)
                    current_ids.extend(level_ids)
                    end_offset = len(current_ids)
                    level_spans.append((start_offset, end_offset))
                    text_parts.append(level_text)

                    end_tok_id = self.special_token_ids["<|level_end|>"]
                    end_idx = [idx_pos for idx_pos, tid in enumerate(level_ids) if tid == end_tok_id]
                    if not end_idx:
                        level_indices.append(len(current_ids) - 1)
                    else:
                        level_indices.append(start_offset + end_idx[-1])

                score_footer = f"{d_score_end}\n"
                footer_ids = self.tokenizer.encode(score_footer, add_special_tokens=False, return_tensors=None)
                start_offset = len(current_ids)
                current_ids.extend(footer_ids)
                text_parts.append(score_footer)

                end_tok_id = self.special_token_ids["<|score_end|>"]
                end_idx = [idx for idx, tid in enumerate(footer_ids) if tid == end_tok_id]
                query_idx = start_offset + end_idx[-1] if end_idx else len(current_ids) - 1

                question_spans_meta[q_id] = QuestionSpans(
                    question_id=q_id,
                    question_type="score",
                    query_token_idx=query_idx,
                    option_keys=level_keys,
                    option_token_indices=level_indices,
                    option_token_spans=level_spans,
                )

        full_text = "".join(text_parts)
        input_ids_tensor = torch.tensor([current_ids], dtype=torch.long)
        attention_mask_tensor = torch.ones_like(input_ids_tensor)

        return PackedSequence(
            text=full_text,
            input_ids=input_ids_tensor,
            attention_mask=attention_mask_tensor,
            questions=question_spans_meta,
        )


_SEQUENTIAL_DEPENDENCY_MARKERS: tuple[str, ...] = (
    "previous question",
    "prior question",
    "previous step",
    "prior step",
    "above decision",
    "earlier question",
)

_TOOL_INVOCATION_PREFIXES: tuple[str, ...] = (
    "tool_",
    "func_",
    "call_",
    "action_",
)

_TOOL_INVOCATION_MARKERS: tuple[str, ...] = (
    "invoke this tool",
    "do not invoke this tool",
    "call this function",
    "do not call this function",
    "func_description",
    "tool_call",
)

_GENERIC_BOOLEAN_KEYS: frozenset[str] = frozenset({"yes", "no", "true", "false", "0", "1"})

_BOILERPLATE_BOOLEAN_DESCRIPTIONS: frozenset[str] = frozenset(
    {
        "yes",
        "no",
        "true",
        "false",
        "yes — affirmative, the condition holds.",
        "no — negative, the condition does not hold.",
        "invoke this tool.",
        "do not invoke this tool.",
        "call this function.",
        "do not call this function.",
    }
)


def has_substantive_options(cleaned_options: list[tuple[str, str]], max_prefix_options: int = 20) -> bool:
    """Returns True if options represent substantive candidate choices rather than boilerplate booleans.

    Substantive options have letter/semantic labels or non-boilerplate descriptions, and moderate
    cardinality (<= max_prefix_options so large-K catalogs are not duplicated in the prefix).
    Boilerplate binary options (yes/no, true/false, 0/1 with generic boolean or tool-gate descriptions)
    return False so they are never repeated as Candidate Options headers.
    """
    if not cleaned_options or len(cleaned_options) > max_prefix_options:
        return False
    keys_low = {str(k).strip().lower() for k, _ in cleaned_options}
    if keys_low <= _GENERIC_BOOLEAN_KEYS:
        descs_low = {str(d).strip().lower() for _, d in cleaned_options}
        if descs_low <= _BOILERPLATE_BOOLEAN_DESCRIPTIONS or len(cleaned_options) == 2:
            return False
    return True


def are_questions_independent(state: Any, questions: Mapping[str, Question]) -> bool:
    """Determines from request structure whether multi-question evaluation can branch per question.

    In ClassOne's API (`DecisionRequest(state, questions)`), questions $q_1, \\dots, q_N$ are
    conditionally independent given `state` if `state` holds the complete world context and each
    question queries `state` independently without depending on sibling questions.

    Conditional independence fails (returning `False`, keeping all questions in a single packed pass)
    in three structural cases:
    1. Single-question requests (`len(questions) <= 1`): no branching needed.
    2. Explicit cross-question or sequential chain references: a question's instructions or criteria
       reference another question ID in `questions` or contain sequential chain markers.
    3. Coupled candidate-pool or tool-invocation routing:
       (a) Question IDs or instructions/criteria represent callable tool/function invocations
           (e.g., `tool_*` / `func_*` prefixes or tool-invocation criteria), where deciding whether
           to invoke tool $T_i$ requires seeing competing tools $T_{j \\neq i}$ in the request; OR
       (b) All questions are binary gates (`NoulQuestion` or 2-option `ChoiceQuestion`) and either:
           - All questions use generic boolean keys (`yes`/`no`, `true`/`false`, `0`/`1`) representing
             a multi-label decomposition of a candidate pool;
           - Any question's `instructions` is a structured candidate object (`dict`/`list`); or
           - The aggregate serialized length of all question instructions exceeds the serialized `state`
             length (`sum(len(inst_i)) > len(state_str)`), indicating that the candidate pool or
             correlated signal set lives in `questions` rather than `state`.
    """
    if len(questions) <= 1:
        return False

    state_str = state.strip() if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    q_ids = list(questions.keys())
    id_patterns = [
        (qid, re.compile(rf"(?<![A-Za-z0-9_]){re.escape(qid)}(?![A-Za-z0-9_])"))
        for qid in q_ids
        if len(qid) >= 3
    ]

    all_binary = True
    all_generic_boolean = True
    has_structured_candidate_inst = False
    total_inst_chars = 0

    for q_id, q in questions.items():
        q_id_low = str(q_id).strip().lower()
        if q_id_low.startswith(_TOOL_INVOCATION_PREFIXES):
            return False

        raw_inst = getattr(q, "instructions", None)
        if isinstance(raw_inst, (dict, list)):
            has_structured_candidate_inst = True
        inst_text = ClassOnePromptBuilder._serialize_text_field(raw_inst)
        crit_text = ClassOnePromptBuilder._serialize_text_field(getattr(q, "criteria", None))
        combined_text = f"{inst_text}\n{crit_text}"
        combined_low = combined_text.lower()

        total_inst_chars += len(inst_text)

        if any(marker in combined_low for marker in _SEQUENTIAL_DEPENDENCY_MARKERS):
            return False
        if any(marker in combined_low for marker in _TOOL_INVOCATION_MARKERS):
            return False

        for other_id, pat in id_patterns:
            if other_id != q_id and pat.search(combined_text):
                return False

        if isinstance(q, NoulQuestion):
            pass
        elif isinstance(q, ChoiceQuestion) and len(q.criteria) == 2:
            keys_low = {str(k).strip().lower() for k in q.criteria}
            if not (keys_low <= _GENERIC_BOOLEAN_KEYS):
                all_generic_boolean = False
        else:
            all_binary = False
            all_generic_boolean = False

    if all_binary:
        if all_generic_boolean or has_structured_candidate_inst or total_inst_chars > len(state_str):
            return False

    return True

