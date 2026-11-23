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
    GEMMA4_CODEBOOK_V1_PATH,
    ChatDecisionPromptBuilder,
    CodebookLMHead,
    build_gemma4_single_token_labels,
    build_single_token_labels,
    compute_lm_letter_logits,
    detect_model_family,
    evaluate_stage2_request,
    load_codebook_lm_head,
    strip_unused_multimodal_towers,
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


def test_detect_model_family_and_strip_unused_multimodal_towers() -> None:
    tok_qwen = _build_chat_test_tokenizer()
    assert detect_model_family(tok_qwen) == "qwen"
    assert detect_model_family({"model_type": "qwen3_5"}) == "qwen"
    assert detect_model_family({"model_type": "gemma4", "text_config": {"model_type": "gemma4_text"}}) == "gemma4"
    assert detect_model_family(None) == "default"

    class _DummyGemmaBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.language_model = torch.nn.Linear(4, 4)
            self.vision_tower = torch.nn.Linear(4, 4)
            self.audio_tower = torch.nn.Linear(4, 4)
            self.embed_vision = torch.nn.Linear(4, 4)
            self.embed_audio = torch.nn.Linear(4, 4)

    bb = _DummyGemmaBackbone()
    stripped = strip_unused_multimodal_towers(bb, clear_cuda_cache=False)
    assert set(stripped) == {"vision_tower", "audio_tower", "embed_vision", "embed_audio"}
    assert bb.vision_tower is None
    assert bb.audio_tower is None
    assert isinstance(bb.language_model, torch.nn.Linear)


def test_gemma4_codebook_v1_determinism_and_softcap_readout() -> None:
    from pathlib import Path
    from safetensors.torch import save_file

    assert GEMMA4_CODEBOOK_V1_PATH.exists(), f"Missing versioned codebook: {GEMMA4_CODEBOOK_V1_PATH}"
    payload = json.loads(GEMMA4_CODEBOOK_V1_PATH.read_text(encoding="utf-8"))
    entries = payload["entries"]
    assert payload["max_k"] == 180 and len(entries) == 180
    assert payload["leading_space_collisions"] == 0

    yes_id = int(payload["yes_token"]["token_id"])
    no_id = int(payload["no_token"]["token_id"])
    ids = [int(e["token_id"]) for e in entries]
    labels = [str(e["label"]) for e in entries]
    assert len(set(ids)) == 180
    assert len(set(labels)) == 180
    assert all(len(lbl) == 1 and lbl.isalnum() for lbl in labels)
    assert yes_id not in ids and no_id not in ids and yes_id != no_id

    # Verify CodebookLMHead & compute_lm_letter_logits softcapping parity + gradients
    torch.manual_seed(7)
    w_rows = torch.randn(6, 16, dtype=torch.bfloat16) * 2.5
    row_ids = [10, 11, 12, 13, 14, 15]
    cb_soft = CodebookLMHead(w_rows, row_ids, final_logit_softcapping=30.0)
    h_q = torch.randn(16, dtype=torch.bfloat16, requires_grad=True)
    probs, logits = compute_lm_letter_logits(h_q, cb_soft, [10, 12, 14], scale=1.0)
    raw = torch.nn.functional.linear(h_q.detach().to(w_rows.dtype), w_rows[[0, 2, 4]])
    expected = (torch.tanh(raw / 30.0) * 30.0).float()
    assert float(torch.max(torch.abs(logits - expected)).item()) < 1e-6
    (-torch.log(probs[0] + 1e-8)).backward()
    assert h_q.grad is not None and torch.isfinite(h_q.grad).all()

    # Verify load_codebook_lm_head allows tied gemma4 embed_tokens and extracts softcapping
    tok = _build_chat_test_tokenizer()
    chat_builder = ChatDecisionPromptBuilder(tok, max_k=8)
    full_w = torch.randn(len(tok), 16, dtype=torch.bfloat16)
    with tempfile.TemporaryDirectory() as tmpdir:
        save_file({"model.language_model.embed_tokens.weight": full_w}, os.path.join(tmpdir, "model.safetensors"))
        gemma_cfg = {
            "model_type": "gemma4",
            "tie_word_embeddings": True,
            "text_config": {"model_type": "gemma4_text", "tie_word_embeddings": True, "final_logit_softcapping": 30.0},
        }
        with open(os.path.join(tmpdir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(gemma_cfg, f)
        loaded = load_codebook_lm_head(tmpdir, chat_builder, device="cpu", dtype=torch.bfloat16)
        assert loaded is not None
        assert loaded.final_logit_softcapping == 30.0
        _, z_cb = compute_lm_letter_logits(h_q.detach(), loaded, [13, 14, 15, 11], scale=1.0)
        _, z_ref = compute_lm_letter_logits(
            h_q.detach(), full_w, [13, 14, 15, 11], scale=1.0, final_logit_softcapping=30.0
        )
        assert float(torch.max(torch.abs(z_cb - z_ref)).item()) < 1e-6

    # Live Gemma 4 E2B tokenizer round-trip check when locally cached
    gemma_snap = Path(
        r"E:\huggingface\hub\models--google--gemma-4-e2b-it\snapshots\3e22461f65e89153144f8adb70e3b8c2cc9845a7"
    )
    if gemma_snap.exists():
        from transformers import AutoTokenizer

        tok_gemma = AutoTokenizer.from_pretrained(str(gemma_snap), local_files_only=True)
        assert detect_model_family(tok_gemma) == "gemma4"
        cb_a = build_gemma4_single_token_labels(tok_gemma, max_k=180)
        cb_b = build_gemma4_single_token_labels(tok_gemma, max_k=180)
        assert cb_a == cb_b
        assert cb_a == [(str(e["label"]), int(e["token_id"])) for e in entries]
        gb = ChatDecisionPromptBuilder(tok_gemma, max_k=180)
        assert gb.yes_no_ids == [yes_id, no_id]
        assert gb.assistant_tail == "<turn|>\n<|turn>model\n"
        packed = gb.pack_question(
            state="Policy: allow refunds within 30 days.",
            q_id="q1",
            question=ChoiceQuestion(instructions="Choose outcome", criteria={"allow": "Allow refund", "deny": "Deny"}),
        )
        assert packed.label_token_ids["q1"] == [entries[0]["token_id"], entries[1]["token_id"]]


def test_gemma4_from_backbone_preserves_fp32_inv_freq_and_heads() -> None:
    from types import SimpleNamespace

    class _DummyRotary(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("full_attention_inv_freq", torch.tensor([1.0, 0.5], dtype=torch.float32), persistent=False)
            self.register_buffer("sliding_attention_inv_freq", torch.tensor([1.0, 0.25], dtype=torch.float32), persistent=False)

    class _DummyGemmaBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(
                model_type="gemma4",
                text_config=SimpleNamespace(model_type="gemma4_text", hidden_size=16, max_position_embeddings=256),
            )
            self.language_model = torch.nn.Linear(16, 16)
            self.rotary_emb = _DummyRotary()

    bb = _DummyGemmaBackbone()
    co = ClassOneModel.from_backbone(bb, device="cpu", torch_dtype=torch.bfloat16)
    assert co.backbone.language_model.weight.dtype == torch.bfloat16
    assert co.backbone.rotary_emb.full_attention_inv_freq.dtype == torch.float32
    assert co.backbone.rotary_emb.sliding_attention_inv_freq.dtype == torch.float32
    assert next(co.noul_head.parameters()).dtype == torch.float32
    assert next(co.choice_head.parameters()).dtype == torch.float32
    assert next(co.score_head.parameters()).dtype == torch.float32

    # Stripped standalone Gemma4TextModel (model_type='gemma4_text') checks
    from classone.stage2_readout import _extract_final_logit_softcapping

    class _DummyGemmaTextBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(
                model_type="gemma4_text",
                hidden_size=16,
                max_position_embeddings=256,
                final_logit_softcapping=30.0,
            )
            self.embed_tokens = torch.nn.Embedding(32, 16)
            self.rotary_emb = _DummyRotary()

    text_bb = _DummyGemmaTextBackbone()
    assert detect_model_family(text_bb) == "gemma4"
    assert _extract_final_logit_softcapping(text_bb.config) == 30.0
    co_text = ClassOneModel.from_backbone(text_bb, device="cpu", torch_dtype=torch.bfloat16)
    assert co_text.backbone.embed_tokens.weight.dtype == torch.bfloat16
    assert co_text.backbone.rotary_emb.full_attention_inv_freq.dtype == torch.float32
    assert next(co_text.noul_head.parameters()).dtype == torch.float32

    # Custom codebook_path forwarding in ChatDecisionPromptBuilder
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocab = {"<unk>": 0, "Yes": 1, "No": 2, "Z": 3, "Q": 4, "user": 5, "assistant": 6}
    tk = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = Whitespace()
    fast_tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>")
    fast_tok.chat_template = (
        "{% for m in messages %}<|turn>{{ m['role'] }}\n{{ m['content'] }}"
        "{% if loop.last and add_generation_prompt %}<turn|>\n<|turn>model\n{% else %}<turn|>\n{% endif %}{% endfor %}"
    )
    assert detect_model_family(fast_tok) == "gemma4"
    with tempfile.TemporaryDirectory() as tmpdir:
        cb_file = os.path.join(tmpdir, "custom_cb.json")
        with open(cb_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "entries": [
                        {"label": "Z", "token_id": 3},
                        {"label": "Q", "token_id": 4},
                    ]
                },
                f,
            )
        cb_builder = ChatDecisionPromptBuilder(fast_tok, max_k=2, codebook_path=cb_file)
        assert cb_builder.codebook == [("Z", 3), ("Q", 4)]
        assert cb_builder.yes_no_ids == [1, 2]


def test_gemma4_model_parallel_userdict_kv_sharing_and_fp32_inv_freq() -> None:
    from collections import UserDict
    from types import SimpleNamespace

    dev0 = "cuda:0" if torch.cuda.device_count() >= 2 else "cpu"
    dev1 = "cuda:1" if torch.cuda.device_count() >= 2 else "cpu"

    class _DummyRotary(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("full_attention_inv_freq", torch.tensor([1.0, 0.5], dtype=torch.float32), persistent=False)

    class _DummyGemmaLayer(torch.nn.Module):
        def __init__(self, idx: int, hidden_size: int) -> None:
            super().__init__()
            self.idx = idx
            self.proj = torch.nn.Linear(hidden_size, hidden_size)

        def forward(self, x: torch.Tensor, shared_kv_states=None, **kwargs) -> torch.Tensor:
            if shared_kv_states is not None:
                if self.idx == 0:
                    shared_kv_states["kv_0"] = torch.ones(1, device=x.device, dtype=x.dtype)
                elif self.idx >= 2 and "kv_0" in shared_kv_states:
                    assert shared_kv_states["kv_0"].device == x.device
            return x + 0.1 * torch.tanh(self.proj(x))

    class _DummyGemmaLanguageModel(torch.nn.Module):
        def __init__(self, vocab_size: int = 32, hidden_size: int = 16) -> None:
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
            self.rotary_emb = _DummyRotary()
            self.layers = torch.nn.ModuleList([_DummyGemmaLayer(i, hidden_size) for i in range(4)])
            self.norm = torch.nn.LayerNorm(hidden_size)

        def get_input_embeddings(self):
            return self.embed_tokens

        def forward(self, input_ids, attention_mask=None, **kwargs):
            h = self.embed_tokens(input_ids)
            shared_kv = UserDict()
            shared_id = id(shared_kv)
            for layer in self.layers:
                h = layer(h, shared_kv_states=shared_kv, attention_mask=attention_mask)
            assert id(shared_kv) == shared_id
            return (self.norm(h),)

    class _DummyGemmaComposite(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(
                model_type="gemma4",
                text_config=SimpleNamespace(model_type="gemma4_text", hidden_size=16, max_position_embeddings=256),
            )
            self.language_model = _DummyGemmaLanguageModel()
            self.vision_tower = torch.nn.Linear(4, 4)
            self.audio_tower = torch.nn.Linear(4, 4)
            self.embed_vision = torch.nn.Linear(4, 4)
            self.embed_audio = torch.nn.Linear(4, 4)

        def get_input_embeddings(self):
            return self.language_model.embed_tokens

        def forward(self, input_ids, attention_mask=None, **kwargs):
            return self.language_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    bb = _DummyGemmaComposite()
    co = ClassOneModel.from_backbone(
        bb,
        torch_dtype=torch.bfloat16,
        model_parallel=True,
        mp_devices=(dev0, dev1),
        mp_split_layer=1,
        head_dtype=torch.float32,
    )
    assert co.backbone.vision_tower is None
    assert co.backbone.audio_tower is None
    assert co.backbone.language_model.rotary_emb.full_attention_inv_freq.dtype == torch.float32
    assert next(co.backbone.language_model.layers[0].parameters()).device == torch.device(dev0)
    assert next(co.backbone.language_model.layers[1].parameters()).device == torch.device(dev1)
    assert next(co.noul_head.parameters()).device == torch.device(dev1)

    co.train()
    inp = torch.tensor([[1, 2, 3]], dtype=torch.long)
    mask = torch.ones_like(inp)
    hs = co.extract_hidden_states(inp, mask)
    assert hs.device == torch.device(dev1)
    prob, _ = co.noul_head(hs[0, -1], return_logits=True)
    prob.backward()
    g0 = co.backbone.language_model.layers[0].proj.weight.grad
    g3 = co.backbone.language_model.layers[3].proj.weight.grad
    assert g0 is not None and torch.isfinite(g0).all() and float(g0.abs().sum()) > 0
    assert g3 is not None and torch.isfinite(g3).all() and float(g3.abs().sum()) > 0
