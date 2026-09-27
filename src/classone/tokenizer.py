"""Prompt formatting, special tokens, and token span tracking for Jev.

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

# Reserved delimiter tokens for Jev decision architecture
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
        # Add special tokens if not already present
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
        state_str = self.serialize_state(state)

        # Assemble full text representation
        text_parts: list[str] = [f"<|state_start|>\n{state_str}\n<|state_end|>\n"]

        question_spans_meta: dict[str, QuestionSpans] = {}

        # Tokenize state prefix first
        prefix_ids = self.tokenizer.encode(
            text_parts[0],
            add_special_tokens=True,
            return_tensors=None,
        )
        current_ids: list[int] = list(prefix_ids)

        for q_id, q in questions.items():
            if isinstance(q, NoulQuestion):
                q_text = f"<|noul_start|>\nQuestion ID: {q_id}\nInstruction: {q.instructions}\n<|noul_end|>\n"
                seg_ids = self.tokenizer.encode(q_text, add_special_tokens=False, return_tensors=None)
                start_offset = len(current_ids)
                current_ids.extend(seg_ids)
                text_parts.append(q_text)

                # Locate <|noul_end|> token position in this segment
                end_tok_id = self.special_token_ids["<|noul_end|>"]
                end_indices = [idx for idx, tid in enumerate(seg_ids) if tid == end_tok_id]
                if not end_indices:
                    # Fallback to the last token in segment
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
                choice_header = f"<|choice_start|>\nQuestion ID: {q_id}\nInstruction: {q.instructions}\n"
                header_ids = self.tokenizer.encode(choice_header, add_special_tokens=False, return_tensors=None)
                current_ids.extend(header_ids)
                text_parts.append(choice_header)

                opt_keys: list[str] = []
                opt_indices: list[int] = []

                for opt_key, opt_desc in q.criteria.items():
                    opt_keys.append(opt_key)
                    opt_text = f"<|opt_start|>\nKey: {opt_key}\nCriteria: {opt_desc}\n<|opt_end|>\n"
                    opt_ids = self.tokenizer.encode(opt_text, add_special_tokens=False, return_tensors=None)
                    start_offset = len(current_ids)
                    current_ids.extend(opt_ids)
                    text_parts.append(opt_text)

                    end_tok_id = self.special_token_ids["<|opt_end|>"]
                    end_idx = [idx for idx, tid in enumerate(opt_ids) if tid == end_tok_id]
                    if not end_idx:
                        opt_indices.append(len(current_ids) - 1)
                    else:
                        opt_indices.append(start_offset + end_idx[-1])

                choice_footer = "<|choice_end|>\n"
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
                )

            elif isinstance(q, ScoreQuestion):
                score_header = f"<|score_start|>\nQuestion ID: {q_id}\nInstruction: {q.instructions}\n"
                header_ids = self.tokenizer.encode(score_header, add_special_tokens=False, return_tensors=None)
                current_ids.extend(header_ids)
                text_parts.append(score_header)

                level_keys: list[str] = []
                level_indices: list[int] = []

                for idx, level_desc in enumerate(q.criteria):
                    level_key = str(idx + 1)
                    level_keys.append(level_key)
                    level_text = f"<|level_start|>\nLevel: {level_key}\nCriteria: {level_desc}\n<|level_end|>\n"
                    level_ids = self.tokenizer.encode(level_text, add_special_tokens=False, return_tensors=None)
                    start_offset = len(current_ids)
                    current_ids.extend(level_ids)
                    text_parts.append(level_text)

                    end_tok_id = self.special_token_ids["<|level_end|>"]
                    end_idx = [idx_pos for idx_pos, tid in enumerate(level_ids) if tid == end_tok_id]
                    if not end_idx:
                        level_indices.append(len(current_ids) - 1)
                    else:
                        level_indices.append(start_offset + end_idx[-1])

                score_footer = "<|score_end|>\n"
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
