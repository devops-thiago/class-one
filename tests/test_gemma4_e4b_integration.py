"""Integration tests verifying ClassOne integration with Gemma 4 E4B architecture (hidden_size=2560)."""

import pytest
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from classone.modeling.configuration_classone import ClassOneConfig
from classone.modeling.modeling_classone import ClassOneModel
from classone.schemas import ChoiceQuestion, NoulQuestion, ScoreQuestion
from classone.tokenizer import ClassOnePromptBuilder


class MockGemma4TextConfig:
    def __init__(self, hidden_size=2560, vocab_size=500):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_hidden_layers = 2
        self.num_attention_heads = 8
        self.num_key_value_heads = 2
        self.intermediate_size = 10240


class MockGemma4Backbone(nn.Module):
    """Simulates a Gemma 4 E4B backbone with hidden_size=2560."""

    def __init__(self, text_config):
        super().__init__()
        self.config = text_config
        self.embed = nn.Embedding(text_config.vocab_size, text_config.hidden_size)
        self.linear = nn.Linear(text_config.hidden_size, text_config.hidden_size)

    def forward(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
        x = self.embed(input_ids)
        h = self.linear(x)

        class MockOutput:
            def __init__(self, last_hidden_state):
                self.last_hidden_state = last_hidden_state

        return MockOutput(h)


@pytest.fixture
def e4b_tokenizer():
    vocab = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
    }
    for i in range(100):
        vocab[f"tok_{i}"] = 4 + i

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


def test_gemma4_e4b_head_binding_and_execution(e4b_tokenizer):
    """Verifies that ClassOne decision heads bind seamlessly to Gemma 4 E4B's 2560 hidden dimension."""
    builder = ClassOnePromptBuilder(e4b_tokenizer)

    t_cfg = MockGemma4TextConfig(hidden_size=2560, vocab_size=len(e4b_tokenizer))
    backbone = MockGemma4Backbone(t_cfg)

    cfg = ClassOneConfig(
        base_model_name_or_path="google/gemma-4-E4B-it",
        hidden_size=2560,
    )
    model = ClassOneModel(config=cfg, backbone=backbone)

    # 1. Verify head projection dimensions
    assert model.noul_head.net[0].in_features == 2560
    assert model.choice_head.q_proj[0].in_features == 2560
    assert model.choice_head.k_proj[0].in_features == 2560
    assert model.score_head.choice_evaluator.q_proj[0].in_features == 2560

    # 2. Pack multi-primitive questions
    state = "Enterprise policy: contract renewal requires executive CFO sign-off above $50k."
    questions = {
        "is_cfo_required": NoulQuestion(instructions="Does this renewal require CFO sign-off?"),
        "approval_tier": ChoiceQuestion(
            instructions="Select required approval tier:",
            criteria={
                "director": "Tier 1: Director approval",
                "cfo": "Tier 2: CFO executive approval",
            },
        ),
        "priority": ScoreQuestion(
            instructions="Rate urgency from low to critical:",
            criteria=["routine", "elevated", "critical"],
        ),
    }

    packed = builder.pack(state, questions)

    # 3. Single-forward pass evaluation
    with torch.no_grad():
        answers = model.evaluate_packed(packed)

    assert "is_cfo_required" in answers
    assert "approval_tier" in answers
    assert "priority" in answers

    # Verify Noul
    assert 0.0 <= answers["is_cfo_required"].noul <= 1.0

    # Verify Choice
    choice_ans = answers["approval_tier"]
    assert choice_ans.choice in ["director", "cfo"]
    assert 0.0 <= choice_ans.confidence <= 1.0
    assert choice_ans.is_decisive in [True, False]
    assert 0.0 <= choice_ans.margin <= 1.0

    # Verify Score
    score_ans = answers["priority"]
    assert 1.0 <= score_ans.score <= 3.0
    assert 0.0 <= score_ans.confidence <= 1.0
