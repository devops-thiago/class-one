"""Unit tests for Stage 2 chat decision prompt building and LM-head letter readout."""

from __future__ import annotations

import json
import os
import tempfile

import torch
from fastapi.testclient import TestClient
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from classone.modeling.configuration_classone import ClassOneConfig
from classone.modeling.modeling_classone import ClassOneModel
from classone.schemas import ChoiceQuestion, NoulQuestion, ScoreQuestion
from classone.server import app as server_app
from classone.stage2_readout import (
    ChatDecisionPromptBuilder,
    build_single_token_labels,
    compute_lm_letter_logits,
    evaluate_stage2_request,
)
from classone.tokenizer import ClassOnePromptBuilder


def _build_chat_test_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {
        "[UNK]": 0,
        "[PAD]": 1,
        "<|im_start|>": 2,
        "<|im_end|>": 3,
        "<think>": 4,
        "</think>": 5,
        "user": 6,
        "assistant": 7,
        "Context:": 8,
        "Task:": 9,
        "Options:": 10,
        "Yes": 11,
        "No": 12,
        "A": 13,
        "B": 14,
        "C": 15,
        "D": 16,
        "a": 17,
        "b": 18,
        "0": 19,
        "1": 20,
        "state": 21,
        "first": 22,
        "second": 23,
    }
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    fast_tok = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]")
    fast_tok.chat_template = (
        "{% for message in messages %}"
        "<|im_start|> {{ message['role'] }} {{ message['content'] }} <|im_end|> "
        "{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|> assistant <think> </think>{% endif %}"
    )
    return fast_tok


def test_single_token_codebook_and_chat_spans() -> None:
    tok = _build_chat_test_tokenizer()
    labels = build_single_token_labels(tok, max_k=6)
    assert [lbl for lbl, _ in labels[:4]] == ["A", "B", "C", "D"]

    builder = ChatDecisionPromptBuilder(tok, max_k=8)

    q_choice = ChoiceQuestion(
        instructions="Pick the first option.",
        criteria={"k1": "first choice", "k2": "second choice"},
    )
    packed_choice = builder.pack_question(state="state", q_id="q1", question=q_choice)
    span_c = packed_choice.questions["q1"]
    assert span_c.option_keys == ["k1", "k2"]
    assert packed_choice.label_strings["q1"] == ["A", "B"]
    assert packed_choice.label_token_ids["q1"] == [13, 14]
    assert len(span_c.option_token_indices) == 2
    assert span_c.query_token_idx == packed_choice.input_ids.shape[1] - 1

    q_noul = NoulQuestion(instructions="Does condition hold?")
    packed_noul = builder.pack_question(state="state", q_id="n1", question=q_noul)
    assert packed_noul.label_token_ids["n1"] == [11, 12]

    q_score = ScoreQuestion(instructions="Rate severity", criteria=["low", "high"])
    packed_score = builder.pack_question(state="state", q_id="s1", question=q_score)
    assert packed_score.questions["s1"].option_keys == ["1", "2"]


def test_compute_lm_letter_logits_and_head_shape_matching() -> None:
    config = ClassOneConfig(hidden_size=64, head_hidden_size=32)
    model = ClassOneModel(config)

    embed_w = torch.randn(32, 64, dtype=torch.bfloat16)
    h_q = torch.randn(64, dtype=torch.bfloat16, requires_grad=True)
    probs, logits = compute_lm_letter_logits(h_q, embed_w, [13, 14, 15], scale=1.0)
    assert probs.shape == (3,)
    assert abs(float(probs.sum().item()) - 1.0) < 1e-5
    loss = -torch.log(probs[1] + 1e-8)
    loss.backward()
    assert h_q.grad is not None and torch.isfinite(h_q.grad).all()

    with tempfile.TemporaryDirectory() as tmpdir:
        heads_file = os.path.join(tmpdir, "classone_heads.pt")
        torch.save(
            {
                "noul_head": model.noul_head.state_dict(),
                "choice_head": {
                    **model.choice_head.state_dict(),
                    "unexpected_extra": torch.zeros(123),
                },
                "score_head": model.score_head.state_dict(),
            },
            heads_file,
        )
        raw = torch.load(heads_file, map_location="cpu", weights_only=True)
        dst = model.choice_head.state_dict()
        matched = {k: v for k, v in raw["choice_head"].items() if k in dst and dst[k].shape == v.shape}
        assert "unexpected_extra" not in matched
        assert len(matched) == len(dst)


def test_dict_criteria_and_instructions_parity() -> None:
    tok = _build_chat_test_tokenizer()
    builder = ChatDecisionPromptBuilder(tok, max_k=8)
    raw_inst = {"goal": "pick", "domain": "billing"}
    q = ChoiceQuestion(
        instructions=raw_inst,
        criteria={"k1": {"what": "first choice", "extra": 1}, "k2": {"what": "second choice"}},
    )
    packed = builder.pack_question(state="state", q_id="q1", question=q)
    assert json.dumps(raw_inst) in packed.text
    assert "A. first choice\n" in packed.text
    assert "B. second choice\n" in packed.text


def test_evaluate_stage2_request_and_server_routing(monkeypatch) -> None:
    tok = _build_chat_test_tokenizer()
    chat_builder = ChatDecisionPromptBuilder(tok, max_k=8)
    prompt_builder = ClassOnePromptBuilder(tok)

    config = ClassOneConfig(vocab_size=len(tok), hidden_size=64, head_hidden_size=32)
    model = ClassOneModel(config)

    questions = {
        "c_1pass": ChoiceQuestion(
            instructions="Pick first",
            criteria={"x1": "first", "x2": "second"},
        ),
        "c_m2": ChoiceQuestion(
            instructions="Pick among 4",
            criteria={"o1": "first", "o2": "second", "o3": "first", "o4": "second"},
        ),
        "n_check": NoulQuestion(instructions="Check condition"),
        "s_rate": ScoreQuestion(instructions="Rate", criteria=["first", "second"]),
    }

    answers, tokens_m2 = evaluate_stage2_request(
        model,
        chat_builder,
        state="state",
        questions=questions,
        m2_k_threshold=4,
        m2_top_k=2,
        max_tokens=512,
    )
    assert answers["c_1pass"].choice in {"x1", "x2"}
    assert abs(sum(answers["c_1pass"].probabilities.values()) - 1.0) < 1e-3
    assert answers["c_m2"].choice in {"o1", "o2", "o3", "o4"}
    assert abs(sum(answers["c_m2"].probabilities.values()) - 1.0) < 1e-3
    assert 0.0 <= answers["n_check"].noul <= 1.0
    assert 1.0 <= answers["s_rate"].score <= 2.0

    _, tokens_no_m2 = evaluate_stage2_request(
        model,
        chat_builder,
        state="state",
        questions={"c_m2": questions["c_m2"]},
        m2_k_threshold=64,
        max_tokens=512,
    )
    _, tokens_with_m2 = evaluate_stage2_request(
        model,
        chat_builder,
        state="state",
        questions={"c_m2": questions["c_m2"]},
        m2_k_threshold=4,
        m2_top_k=2,
        max_tokens=512,
    )
    assert tokens_with_m2 > tokens_no_m2

    # FastAPI /v1/decide end-to-end check with Stage 2 chat builder injected
    orig_state = (server_app._model, server_app._prompt_builder, server_app._chat_builder)
    try:
        server_app.set_pipeline(model, prompt_builder, chat_builder=chat_builder)
        monkeypatch.setenv("CLASSONE_EVAL_MODE", "stage2")
        monkeypatch.setenv("CLASSONE_M2_K_THRESHOLD", "4")
        monkeypatch.setenv("CLASSONE_M2_TOP_K", "2")
        monkeypatch.setenv("CLASSONE_MAX_TOKENS_PER_REQUEST", "512")

        client = TestClient(server_app.app)
        resp = client.post(
            "/v1/decide",
            json={
                "model": "v1_merge_0.40_0.60",
                "state": "state",
                "questions": {
                    "q_m2": {
                        "type": "choice",
                        "instructions": "Pick among 4",
                        "criteria": {"o1": "first", "o2": "second", "o3": "first", "o4": "second"},
                    },
                    "q_noul": {"type": "noul", "instructions": "Check condition"},
                },
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["usage"]["input_tokens"] > 0
        assert body["answers"]["q_m2"]["choice"] == answers["c_m2"].choice
        assert body["answers"]["q_noul"]["noul"] == answers["n_check"].noul
    finally:
        server_app.set_pipeline(orig_state[0], orig_state[1], chat_builder=orig_state[2])


def test_sdk_exports_and_stage2_config_propagation() -> None:
    import classone

    assert classone.__version__ == "0.3.0"
    for sym in (
        "ChatDecisionPromptBuilder",
        "ChatPackedQuestion",
        "build_single_token_labels",
        "compute_lm_letter_logits",
        "evaluate_stage2_request",
        "load_stage2_snapshot",
    ):
        assert getattr(classone, sym) is not None

    default_cfg = ClassOneConfig()
    assert default_cfg.eval_mode == "stage1"
    assert default_cfg.m2_k_threshold == 64
    assert default_cfg.m2_top_k == 16
    assert default_cfg.classone_config is None

    tok = _build_chat_test_tokenizer()
    chat_builder = ChatDecisionPromptBuilder(tok, max_k=8)

    class _DummyBackbone(torch.nn.Module):
        def __init__(self, vocab_size: int = 32, hidden_size: int = 64):
            super().__init__()
            self.embed = torch.nn.Embedding(vocab_size, hidden_size)
            self.config = type(
                "_Cfg",
                (),
                {
                    "hidden_size": hidden_size,
                    "classone_config": {
                        "eval_mode": "stage2",
                        "m2_k_threshold": 4,
                        "m2_top_k": 2,
                    },
                },
            )()

        def get_input_embeddings(self):
            return self.embed

        def forward(self, input_ids, attention_mask=None, **kwargs):
            return (self.embed(input_ids),)

    backbone = _DummyBackbone()
    model = ClassOneModel.from_backbone(backbone, device="cpu")
    assert model.config.eval_mode == "stage2"
    assert model.config.m2_k_threshold == 4
    assert model.config.m2_top_k == 2
    assert model.config.classone_config == {
        "eval_mode": "stage2",
        "m2_k_threshold": 4,
        "m2_top_k": 2,
    }

    answers, total_tokens = model.evaluate_request(
        chat_builder,
        state="state",
        questions={
            "c_m2": ChoiceQuestion(
                instructions="Pick among 4",
                criteria={"o1": "first", "o2": "second", "o3": "first", "o4": "second"},
            ),
            "n1": NoulQuestion(instructions="Check condition"),
        },
        max_tokens=512,
    )
    assert total_tokens > 0
    assert answers["c_m2"].choice in {"o1", "o2", "o3", "o4"}
    assert 0.0 <= answers["n1"].noul <= 1.0
