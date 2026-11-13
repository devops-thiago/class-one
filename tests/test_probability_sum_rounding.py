"""Regression tests for GitHub issue #21: rounded probabilities must satisfy sum-to-1 within kit tolerance."""

from __future__ import annotations

import math

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from classone.modeling.configuration_classone import ClassOneConfig
from classone.modeling.modeling_classone import ClassOneModel
from classone.schemas import ChoiceQuestion, ChoiceResult, ScoreQuestion, ScoreResult
from classone.tokenizer import ClassOnePromptBuilder

KIT_SUM_TOLERANCE = 0.01


def _make_tiny_model_and_builder() -> tuple[ClassOneModel, ClassOnePromptBuilder]:
    vocab = {"[UNK]": 0, "[PAD]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]")
    builder = ClassOnePromptBuilder(hf_tok)
    model = ClassOneModel(ClassOneConfig(hidden_size=64, head_hidden_size=32))
    return model, builder


def _assert_valid_distribution(prob_dict: dict[str, float], expected_keys: list[str]) -> None:
    assert list(prob_dict.keys()) == expected_keys
    assert all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in prob_dict.values())
    assert abs(sum(prob_dict.values()) - 1.0) <= KIT_SUM_TOLERANCE


def test_round_4_decimal_math_drifts_past_kit_tolerance_for_255_options() -> None:
    tail = 4.999e-5
    raw = [1.0 - 254 * tail] + [tail] * 254
    naive_sum = sum(round(v, 4) for v in raw)
    assert abs(naive_sum - 1.0) > KIT_SUM_TOLERANCE


def test_emitted_probabilities_sum_to_one_with_255_options(monkeypatch) -> None:
    model, builder = _make_tiny_model_and_builder()
    tail = 4.999e-5
    adversarial = torch.tensor([1.0 - 254 * tail] + [tail] * 254, dtype=torch.float32)
    monkeypatch.setattr(model.choice_head, "forward", lambda h_q, h_opts: adversarial)

    criteria = {f"opt_{i}": f"Criterion {i}" for i in range(255)}
    packed = builder.pack(
        state="Pick one of 255 options",
        questions={"q_255": ChoiceQuestion(instructions="Select best", criteria=criteria)},
        num_pause_tokens=0,
    )
    res = model.evaluate_packed(packed)["q_255"]
    assert isinstance(res, ChoiceResult)
    assert res.choice == "opt_0"
    _assert_valid_distribution(res.probabilities, list(criteria.keys()))
    assert res.probabilities["opt_0"] == max(res.probabilities.values())


def test_emitted_probabilities_sum_to_one_with_two_options(monkeypatch) -> None:
    model, builder = _make_tiny_model_and_builder()
    adversarial = torch.tensor([0.50004, 0.49996], dtype=torch.float32)
    monkeypatch.setattr(model.choice_head, "forward", lambda h_q, h_opts: adversarial)

    criteria = {"left": "Left choice", "right": "Right choice"}
    packed = builder.pack(
        state="Binary decision",
        questions={"q_2": ChoiceQuestion(instructions="Choose one", criteria=criteria)},
        num_pause_tokens=0,
    )
    res = model.evaluate_packed(packed)["q_2"]
    assert isinstance(res, ChoiceResult)
    assert res.choice == "left"
    _assert_valid_distribution(res.probabilities, ["left", "right"])
    assert res.probabilities["left"] >= res.probabilities["right"]


def test_emitted_probabilities_sum_to_one_with_uniform_255(monkeypatch) -> None:
    model, builder = _make_tiny_model_and_builder()
    uniform = torch.full((255,), 1.0 / 255.0, dtype=torch.float32)
    monkeypatch.setattr(model.choice_head, "forward", lambda h_q, h_opts: uniform)

    criteria = {f"k_{i}": f"Option {i}" for i in range(255)}
    packed = builder.pack(
        state="Uniform distribution",
        questions={"q_unif": ChoiceQuestion(instructions="Pick", criteria=criteria)},
        num_pause_tokens=0,
    )
    res = model.evaluate_packed(packed)["q_unif"]
    assert isinstance(res, ChoiceResult)
    _assert_valid_distribution(res.probabilities, list(criteria.keys()))
    assert res.probabilities[res.choice] == max(res.probabilities.values())


def test_emitted_score_probabilities_sum_to_one_when_rounding_drifts(monkeypatch) -> None:
    model, builder = _make_tiny_model_and_builder()
    # Adversarial 10-level distribution where naive round(v, 4) drifts from 1.0
    adversarial = torch.tensor(
        [0.99955] + [4.999e-5] * 9,
        dtype=torch.float32,
    )
    monkeypatch.setattr(
        model.score_head,
        "forward",
        lambda h_q, h_levels: (torch.tensor(1.0, dtype=torch.float32), adversarial),
    )

    levels = [f"level_{i}" for i in range(1, 11)]
    packed = builder.pack(
        state="Score rubric",
        questions={"q_score": ScoreQuestion(instructions="Rate", criteria=levels)},
        num_pause_tokens=0,
    )
    res = model.evaluate_packed(packed)["q_score"]
    assert isinstance(res, ScoreResult)
    _assert_valid_distribution(res.probabilities, [str(i) for i in range(1, 11)])
