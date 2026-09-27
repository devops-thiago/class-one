"""Unit tests for ClassOne model architecture and decision heads."""

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from classone.modeling.configuration_classone import ClassOneConfig
from classone.modeling.modeling_classone import ChoiceHead, ClassOneModel, NoulHead, ScoreHead
from classone.schemas import ChoiceQuestion, NoulQuestion, ScoreQuestion
from classone.tokenizer import ClassOnePromptBuilder


@pytest.fixture
def tokenizer_and_builder():
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
    builder = ClassOnePromptBuilder(fast_tok)
    return fast_tok, builder


def test_noul_head():
    head = NoulHead(hidden_size=64, head_hidden_size=32)
    h_query = torch.randn(64)
    prob = head(h_query)
    assert 0.0 <= prob.item() <= 1.0


def test_choice_head():
    head = ChoiceHead(hidden_size=64, head_hidden_size=32)
    h_query = torch.randn(64)
    h_opts = torch.randn(3, 64)
    probs = head(h_query, h_opts)
    assert probs.shape == (3,)
    assert pytest.approx(probs.sum().item(), abs=1e-5) == 1.0


def test_score_head():
    head = ScoreHead(hidden_size=64, head_hidden_size=32)
    h_query = torch.randn(64)
    h_levels = torch.randn(4, 64)
    score, probs = head(h_query, h_levels)
    assert 1.0 <= score.item() <= 4.0
    assert pytest.approx(probs.sum().item(), abs=1e-5) == 1.0


def test_score_head_half_precision():
    head = ScoreHead(hidden_size=64, head_hidden_size=32).to(dtype=torch.float16)
    h_query = torch.randn(64, dtype=torch.float16)
    h_levels = torch.randn(4, 64, dtype=torch.float16)
    score, _ = head(h_query, h_levels)
    assert score.dtype == torch.float16
    assert 1.0 <= score.item() <= 4.0


def test_classone_model_forward_contract():
    config = ClassOneConfig(hidden_size=64, head_hidden_size=32)
    model = ClassOneModel(config)
    input_ids = torch.randint(0, 1000, (1, 16))
    hidden_states = model(input_ids)
    assert hidden_states.shape == (1, 16, 64)


def test_classone_model_evaluate_packed(tokenizer_and_builder):
    _, builder = tokenizer_and_builder
    config = ClassOneConfig(hidden_size=64, head_hidden_size=32)
    model = ClassOneModel(config)

    state = {"customer": "Alice", "balance": -40.0}
    questions = {
        "overdrawn": NoulQuestion(instructions="Is account overdrawn?"),
        "routing": ChoiceQuestion(
            instructions="Department",
            criteria={"billing": "Billing", "support": "Customer Support"},
        ),
        "satisfaction": ScoreQuestion(
            instructions="Satisfaction rating",
            criteria=["angry", "neutral", "happy"],
        ),
    }

    packed = builder.pack(state=state, questions=questions)
    answers = model.evaluate_packed(packed)

    assert "overdrawn" in answers
    assert answers["overdrawn"].type == "noul"
    assert 0.0 <= answers["overdrawn"].noul <= 1.0

    assert "routing" in answers
    assert answers["routing"].type == "choice"
    assert answers["routing"].choice in ["billing", "support"]
    assert len(answers["routing"].probabilities) == 2
    assert 0.0 <= answers["routing"].confidence <= 1.0

    assert "satisfaction" in answers
    assert answers["satisfaction"].type == "score"
    assert 1.0 <= answers["satisfaction"].score <= 3.0
    assert len(answers["satisfaction"].probabilities) == 3


def test_classone_model_evaluate_batch(tokenizer_and_builder):
    _, builder = tokenizer_and_builder
    config = ClassOneConfig(hidden_size=64, head_hidden_size=32)
    model = ClassOneModel(config)

    p1 = builder.pack(
        state="Customer A",
        questions={"q": NoulQuestion(instructions="Is active?")},
    )
    p2 = builder.pack(
        state="Customer B has a much longer text sequence description.",
        questions={
            "q": NoulQuestion(instructions="Is active?"),
            "category": ChoiceQuestion(
                instructions="Type",
                criteria={"vip": "VIP customer", "standard": "Standard"},
            ),
        },
    )

    batch_answers = model.evaluate_batch([p1, p2])
    assert len(batch_answers) == 2
    assert "q" in batch_answers[0]
    assert batch_answers[0]["q"].type == "noul"
    assert "q" in batch_answers[1]
    assert "category" in batch_answers[1]
    assert batch_answers[1]["category"].choice in ["vip", "standard"]
