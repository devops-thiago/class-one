"""Regression tests for GitHub issue #18: non-string instructions & choice criteria coercion."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from classone.schemas import ChoiceQuestion, NoulQuestion, ScoreQuestion
from classone.server.app import app
from classone.tokenizer import ClassOnePromptBuilder


def _make_dummy_builder() -> ClassOnePromptBuilder:
    vocab = {"[UNK]": 0, "[PAD]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]")
    return ClassOnePromptBuilder(hf_tok)


def test_noul_question_accepts_dict_instructions() -> None:
    q = NoulQuestion(instructions={"query": "Find auth middleware", "doc": "src/auth.py"})
    assert isinstance(q.instructions, dict)


def test_choice_question_accepts_dict_and_list_instructions() -> None:
    q_dict = ChoiceQuestion(
        instructions={"task": "route", "domain": "toolret"},
        criteria={"a": "first", "b": "second"},
    )
    q_list = ChoiceQuestion(
        instructions=["step 1", "step 2"],
        criteria={"a": "first", "b": "second"},
    )
    assert isinstance(q_dict.instructions, dict)
    assert isinstance(q_list.instructions, list)


def test_score_question_accepts_non_string_and_none_instructions() -> None:
    q_dict = ScoreQuestion(instructions={"rubric": "relevance"}, criteria=["low", "high"])
    q_none = ScoreQuestion(instructions=None, criteria=["low", "high"])
    assert q_dict.instructions == {"rubric": "relevance"}
    assert q_none.instructions is None


def test_choice_question_accepts_list_criteria_cfcolor_style() -> None:
    q = ChoiceQuestion(
        instructions="Pick matching color code",
        criteria={"opt_0": ["255", "0", "0"], "opt_1": ["0", "255", "0"]},
    )
    assert q.criteria["opt_0"] == ["255", "0", "0"]


def test_choice_question_accepts_flat_dict_criteria_chessbench_style() -> None:
    q = ChoiceQuestion(
        instructions="Pick best move",
        criteria={
            "e2e4": {"uci": "e2e4", "san": "e4"},
            "d2d4": {"uci": "d2d4", "san": "d4"},
        },
    )
    assert q.criteria["e2e4"]["san"] == "e4"


def test_choice_question_accepts_nested_dict_criteria_pop909_style() -> None:
    q = ChoiceQuestion(
        instructions="Select chord progression",
        criteria={
            "cand_1": {"chord": "Cmaj7", "timing": {"beat": 1, "bar": 2}},
            "cand_2": {"chord": "G7", "timing": {"beat": 3, "bar": 2}},
        },
    )
    assert q.criteria["cand_1"]["timing"]["bar"] == 2


def test_choice_question_accepts_none_and_empty_criteria_phishnchips_style() -> None:
    q = ChoiceQuestion(
        instructions="Classify email",
        criteria={"phishing": None, "benign": ""},
    )
    assert q.criteria["phishing"] is None
    assert q.criteria["benign"] == ""


def test_choice_question_still_rejects_invalid_cardinality_and_empty_key() -> None:
    with pytest.raises(ValidationError):
        ChoiceQuestion(instructions="Too few", criteria={"only_one": None})
    with pytest.raises(ValidationError):
        ChoiceQuestion(instructions="Empty key", criteria={"": "desc", "valid": "desc"})


def test_prompt_builder_preserves_plain_strings_and_coerces_none_to_key() -> None:
    builder = _make_dummy_builder()
    packed = builder.pack(
        state="Email text",
        questions={
            "plain_noul": NoulQuestion(instructions="Is this spam?"),
            "phish": ChoiceQuestion(
                instructions="Select label",
                criteria={"phishing": None, "benign": ""},
            ),
        },
        num_pause_tokens=0,
    )
    assert "Instruction: Is this spam?\n" in packed.text
    assert 'Instruction: "Is this spam?"' not in packed.text
    assert "Criteria: phishing\n" in packed.text
    assert "Criteria: benign\n" in packed.text


def test_prompt_builder_serializes_non_strings_deterministically_with_sorted_keys() -> None:
    builder = _make_dummy_builder()
    q1 = ChoiceQuestion(
        instructions={"b": 1, "a": {"z": 2, "y": 3}},
        criteria={
            "m1": {"uci": "e2e4", "san": "e4"},
            "m2": ["a", "b"],
        },
    )
    q2 = ChoiceQuestion(
        instructions={"a": {"y": 3, "z": 2}, "b": 1},
        criteria={
            "m1": {"san": "e4", "uci": "e2e4"},
            "m2": ["a", "b"],
        },
    )
    p1 = builder.pack(state="Position", questions={"q": q1}, num_pause_tokens=0)
    p2 = builder.pack(state="Position", questions={"q": q2}, num_pause_tokens=0)
    assert p1.text == p2.text
    assert json.dumps({"a": {"y": 3, "z": 2}, "b": 1}, sort_keys=True) in p1.text
    assert '{"san": "e4", "uci": "e2e4"}' in p1.text


def test_prompt_builder_preserves_unicode_and_long_context_preamble_with_dict_instructions() -> None:
    builder = _make_dummy_builder()
    long_state = " ".join(["token"] * 650)
    packed = builder.pack(
        state=long_state,
        questions={
            "unicode_q": ChoiceQuestion(
                instructions={"tarefa": "Classificação"},
                criteria={"jp": {"city": "東京"}, "pt": {"word": "ação"}},
            )
        },
        num_pause_tokens=0,
    )
    assert "Classificação" in packed.text
    assert "東京" in packed.text
    assert "ação" in packed.text
    assert "\\u00e7" not in packed.text
    assert "\\u6771" not in packed.text


@pytest.mark.parametrize("endpoint", ["/v1/systemone", "/v1/decide", "/v1/classone"])
def test_server_endpoints_return_200_for_structured_decision_index_payloads(endpoint: str) -> None:
    client = TestClient(app)
    payload = {
        "model": "classone-qwen3.5-9b",
        "state": {"context": "Sample decision index item"},
        "questions": {
            "toolret_check": {
                "type": "noul",
                "instructions": {"query": "Find SQL tool", "candidate": {"name": "sql_exec"}},
            },
            "chess_move": {
                "type": "choice",
                "instructions": {"fen": "startpos", "side": "white"},
                "criteria": {
                    "e2e4": {"uci": "e2e4", "san": "e4"},
                    "d2d4": {"uci": "d2d4", "san": "d4"},
                },
            },
            "cfcolor_pick": {
                "type": "choice",
                "instructions": None,
                "criteria": {
                    "c1": [255, 0, 0],
                    "c2": [0, 0, 255],
                },
            },
            "phish_label": {
                "type": "choice",
                "instructions": "Detect phishing",
                "criteria": {
                    "phishing": None,
                    "benign": None,
                },
            },
            "pop_rubric": {
                "type": "score",
                "instructions": ["harmonic clarity", {"scale": 3}],
                "criteria": ["poor", "fair", "good"],
            },
        },
    }
    resp = client.post(endpoint, json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body["answers"].keys()) == {"toolret_check", "chess_move", "cfcolor_pick", "phish_label", "pop_rubric"}
