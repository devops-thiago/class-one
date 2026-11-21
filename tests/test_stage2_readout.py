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
    CodebookLMHead,
    build_single_token_labels,
    compute_lm_letter_logits,
    evaluate_stage2_request,
    load_codebook_lm_head,
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


def test_codebook_lm_head_parity_and_gradients() -> None:
    from safetensors.torch import save_file

    tok = _build_chat_test_tokenizer()
    chat_builder = ChatDecisionPromptBuilder(tok, max_k=8)
    vocab_size, hidden_size = len(tok), 64
    full_w = torch.randn(vocab_size, hidden_size, dtype=torch.bfloat16)

    ids = [tid for _, tid in chat_builder.codebook] + [tid for tid in chat_builder.yes_no_ids if tid not in {t for _, t in chat_builder.codebook}]
    slice_w = full_w.index_select(0, torch.tensor(ids, dtype=torch.long)).clone()
    cb_head = CodebookLMHead(slice_w, ids)
    assert sum(p.numel() for p in cb_head.parameters()) == 0  # frozen buffer only

    h_q_ref = torch.randn(hidden_size, dtype=torch.bfloat16, requires_grad=True)
    h_q_cb = h_q_ref.detach().clone().requires_grad_(True)
    query_ids = [13, 14, 15, 11]
    p_ref, z_ref = compute_lm_letter_logits(h_q_ref, full_w, query_ids, scale=1.0)
    p_cb, z_cb = compute_lm_letter_logits(h_q_cb, cb_head, query_ids, scale=1.0)
    assert float(torch.max(torch.abs(z_ref - z_cb)).item()) < 1e-6
    assert float(torch.max(torch.abs(p_ref - p_cb)).item()) < 1e-6

    (-torch.log(p_cb[0] + 1e-8)).backward()
    assert h_q_cb.grad is not None and torch.isfinite(h_q_cb.grad).all()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_file({"lm_head.weight": full_w}, os.path.join(tmpdir, "model.safetensors"))
        cfg_untied = {"tie_word_embeddings": False}
        with open(os.path.join(tmpdir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg_untied, f)
        loaded = load_codebook_lm_head(tmpdir, chat_builder, device="cpu", dtype=torch.bfloat16)
        assert loaded is not None
        _, z_loaded = compute_lm_letter_logits(h_q_cb.detach(), loaded, query_ids, scale=1.0)
        assert float(torch.max(torch.abs(z_ref - z_loaded)).item()) < 1e-6

        with open(os.path.join(tmpdir, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"tie_word_embeddings": True}, f)
        assert load_codebook_lm_head(tmpdir, chat_builder, device="cpu") is None


def test_model_parallel_device_routing_and_backward() -> None:
    tok = _build_chat_test_tokenizer()
    chat_builder = ChatDecisionPromptBuilder(tok, max_k=8)

    class _DummyBlock(torch.nn.Module):
        def __init__(self, hidden_size: int):
            super().__init__()
            self.proj = torch.nn.Linear(hidden_size, hidden_size)

        def forward(self, x: torch.Tensor, **kwargs):
            return x + 0.1 * torch.tanh(self.proj(x))

    class _DummyLayeredBackbone(torch.nn.Module):
        def __init__(self, vocab_size: int = 32, hidden_size: int = 64):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
            self.layers = torch.nn.ModuleList([_DummyBlock(hidden_size) for _ in range(4)])
            self.norm = torch.nn.LayerNorm(hidden_size)
            self.config = type("_Cfg", (), {"hidden_size": hidden_size})()

        def get_input_embeddings(self):
            return self.embed_tokens

        def forward(self, input_ids, attention_mask=None, **kwargs):
            h = self.embed_tokens(input_ids)
            for layer in self.layers:
                h = layer(h, attention_mask=attention_mask)
            return (self.norm(h),)

    # Default model_parallel=False behavior unchanged
    single_model = ClassOneModel.from_backbone(_DummyLayeredBackbone(), device="cpu")
    assert single_model.is_model_parallel is False
    assert single_model.input_device == torch.device("cpu")
    assert single_model.output_device == torch.device("cpu")

    dev0 = "cuda:0" if torch.cuda.device_count() >= 2 else "cpu"
    dev1 = "cuda:1" if torch.cuda.device_count() >= 2 else "cpu"

    backbone = _DummyLayeredBackbone()
    model = ClassOneModel.from_backbone(
        backbone,
        model_parallel=True,
        mp_devices=(dev0, dev1),
        mp_split_layer=2,
        head_dtype=torch.float32,
    )
    assert model.is_model_parallel is True
    assert model.input_device == torch.device(dev0)
    assert model.output_device == torch.device(dev1)
    assert next(model.backbone.layers[0].parameters()).device == torch.device(dev0)
    assert next(model.backbone.layers[2].parameters()).device == torch.device(dev1)
    assert next(model.noul_head.parameters()).device == torch.device(dev1)

    model.train()
    inp = torch.tensor([[21, 22, 23]], dtype=torch.long)
    mask = torch.ones_like(inp)
    hs = model.extract_hidden_states(inp, mask)
    assert hs.device == torch.device(dev1)
    prob, _ = model.noul_head(hs[0, -1], return_logits=True)
    prob.backward()

    g0 = model.backbone.layers[0].proj.weight.grad
    g3 = model.backbone.layers[3].proj.weight.grad
    assert g0 is not None and torch.isfinite(g0).all() and float(g0.abs().sum()) > 0
    assert g3 is not None and torch.isfinite(g3).all() and float(g3.abs().sum()) > 0

    model.lm_readout_head = CodebookLMHead(
        torch.randn(len(chat_builder.codebook), 64, device=dev1, dtype=torch.float32),
        [tid for _, tid in chat_builder.codebook],
    )
    answers, toks = model.evaluate_request(
        chat_builder,
        state="state",
        questions={
            "c1": ChoiceQuestion(instructions="Pick", criteria={"x1": "first", "x2": "second"}),
            "n1": NoulQuestion(instructions="Check"),
        },
        max_tokens=128,
    )
    assert toks > 0
    assert answers["c1"].choice in {"x1", "x2"}
    assert 0.0 <= answers["n1"].noul <= 1.0
