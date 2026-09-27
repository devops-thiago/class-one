"""Integration tests verifying ClassOne integration with Google's Gemma model architecture."""

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GemmaConfig, GemmaModel, PreTrainedTokenizerFast

from classone.modeling.modeling_classone import ClassOneModel
from classone.schemas import ChoiceQuestion, NoulQuestion, ScoreQuestion
from classone.tokenizer import ClassOnePromptBuilder


@pytest.fixture
def gemma_tokenizer():
    vocab = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
    }
    # Add dummy words
    for i in range(50):
        vocab[f"word_{i}"] = 4 + i

    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    return fast_tok


def test_gemma_architecture_end_to_end(gemma_tokenizer):
    """Instantiates a scaled GemmaModel backbone and verifies ClassOne decision execution."""
    builder = ClassOnePromptBuilder(gemma_tokenizer)

    # Lightweight GemmaConfig for rapid unit/integration testing
    gemma_cfg = GemmaConfig(
        vocab_size=len(gemma_tokenizer),
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
    )
    gemma_backbone = GemmaModel(gemma_cfg)

    # Wrap Gemma in ClassOneModel
    classone_gemma = ClassOneModel.from_backbone(
        base_model_name_or_path=gemma_backbone,
        tokenizer=gemma_tokenizer,
        device="cpu",
    )

    state = {
        "user": "Developer",
        "action": "API key creation",
        "rate_limit_exceeded": True,
    }
    questions = {
        "should_throttle": NoulQuestion(instructions="Throttle incoming requests?"),
        "severity": ChoiceQuestion(
            instructions="Incident classification",
            criteria={"info": "Standard activity", "warn": "Warning", "crit": "Critical"},
        ),
        "impact": ScoreQuestion(
            instructions="Impact score",
            criteria=["isolated", "team", "organization"],
        ),
    }

    packed = builder.pack(state=state, questions=questions)
    assert packed.input_ids.shape[0] == 1

    # Run decision pass through Gemma backbone
    answers = classone_gemma.evaluate_packed(packed)

    assert "should_throttle" in answers
    assert answers["should_throttle"].type == "noul"
    assert 0.0 <= answers["should_throttle"].noul <= 1.0

    assert "severity" in answers
    assert answers["severity"].type == "choice"
    assert answers["severity"].choice in ["info", "warn", "crit"]
    assert len(answers["severity"].probabilities) == 3
    assert 0.0 <= answers["severity"].confidence <= 1.0

    assert "impact" in answers
    assert answers["impact"].type == "score"
    assert 1.0 <= answers["impact"].score <= 3.0
    assert len(answers["impact"].probabilities) == 3
