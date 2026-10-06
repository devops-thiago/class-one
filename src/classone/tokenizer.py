"""Prompt formatting, special tokens, and token span tracking for ClassOne.

Serializes state and questions into a single structured sequence, tracking the
exact token index of decision delimiter tokens (<|noul_end|>, <|choice_end|>, etc.)
for single-pass parallel pooling.
"""

from __future__ import annotations

import json
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

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        has_existing_custom = (
            hasattr(tokenizer, "convert_tokens_to_ids")
            and tokenizer.convert_tokens_to_ids("<|state_start|>") is not None
            and tokenizer.convert_tokens_to_ids("<|state_start|>") != getattr(tokenizer, "unk_token_id", -1)
        )
        has_unused = (
            not has_existing_custom
            and hasattr(tokenizer, "convert_tokens_to_ids")
            and tokenizer.convert_tokens_to_ids("<unused0>") is not None
            and tokenizer.convert_tokens_to_ids("<unused0>") != getattr(tokenizer, "unk_token_id", -1)
        )

        if has_unused:
            # Map delimiters to native in-vocab reserved tokens (<unused0>..<unused11>)
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

    def pack(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: dict[str, Question],
    ) -> PackedSequence:
        """Packs state and questions into a single sequence with exact token positions.

        To guarantee exact token index tracking without tokenizer ambiguity, tokens
        are constructed segment-by-segment.
        """
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

        if len(prefix_ids) > 600 and questions:
            instructions_summary = "\n".join(
                f"- [{qid}]: {getattr(q, 'instructions', '').strip()}" for qid, q in questions.items()
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

        for q_id, q in questions.items():
            if isinstance(q, NoulQuestion):
                q_text = f"{d_noul_start}\nQuestion ID: {q_id}\nInstruction: {q.instructions}\n{d_noul_end}\n"
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
                # Choice preamble
                choice_header = f"{d_choice_start}\nQuestion ID: {q_id}\nInstruction: {q.instructions}\n"
                header_ids = self.tokenizer.encode(choice_header, add_special_tokens=False, return_tensors=None)
                current_ids.extend(header_ids)
                text_parts.append(choice_header)

                opt_keys: list[str] = []
                opt_indices: list[int] = []
                opt_spans: list[tuple[int, int]] = []

                for opt_key, opt_desc in q.criteria.items():
                    opt_keys.append(opt_key)
                    opt_text = f"{d_opt_start}\nKey: {opt_key}\nCriteria: {opt_desc}\n{d_opt_end}\n"
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
                score_header = f"{d_score_start}\nQuestion ID: {q_id}\nInstruction: {q.instructions}\n"
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
