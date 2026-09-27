"""Unit tests for Jev prompt builder and token span mapper."""

import pytest
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from classone.schemas import ChoiceQuestion, NoulQuestion, ScoreQuestion
from classone.tokenizer import SPECIAL_TOKENS, ClassOnePromptBuilder


@pytest.fixture
def dummy_tokenizer():
    # Build a tiny local in-memory WordLevel/BPE tokenizer for fast, offline unit testing
    from tokenizers.models import WordLevel

    vocab = {
        "[UNK]": 0,
        "[PAD]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
        "[MASK]": 4,
    }
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    return fast_tok


def test_special_tokens_registration(dummy_tokenizer):
    builder = ClassOnePromptBuilder(dummy_tokenizer)
    for token in SPECIAL_TOKENS:
        assert token in builder.special_token_ids
        assert builder.special_token_ids[token] is not None


def test_pack_all_question_types(dummy_tokenizer):
    builder = ClassOnePromptBuilder(dummy_tokenizer)

    state = {"customer_id": 1234, "issue": "Double charged for subscription."}
    questions = {
        "is_refund": NoulQuestion(instructions="Is this asking for a refund?"),
        "dept": ChoiceQuestion(
            instructions="Target team",
            criteria={"billing": "Billing and fees", "tech": "Bug reports"},
        ),
        "priority": ScoreQuestion(
            instructions="Priority level",
            criteria=["low", "medium", "urgent"],
        ),
    }

    packed = builder.pack(state=state, questions=questions)

    assert packed.input_ids.shape[0] == 1
    assert packed.input_ids.shape[1] > 0
    assert len(packed.questions) == 3

    # Check Noul
    noul_meta = packed.questions["is_refund"]
    assert noul_meta.question_type == "noul"
    assert noul_meta.query_token_idx < packed.input_ids.shape[1]

    # Check Choice
    choice_meta = packed.questions["dept"]
    assert choice_meta.question_type == "choice"
    assert choice_meta.option_keys == ["billing", "tech"]
    assert len(choice_meta.option_token_indices) == 2
    for idx in choice_meta.option_token_indices:
        assert idx < packed.input_ids.shape[1]

    # Check Score
    score_meta = packed.questions["priority"]
    assert score_meta.question_type == "score"
    assert score_meta.option_keys == ["1", "2", "3"]
    assert len(score_meta.option_token_indices) == 3
