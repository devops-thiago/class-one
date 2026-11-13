"""Regression tests for GitHub issue #24: server prompt limit derived from model config."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient

from classone.schemas import NoulResult
from classone.server import app as server_app
from classone.tokenizer import PackedSequence, QuestionSpans

CAPACITY_MARKERS = (
    "context length",
    "maximum context length",
    "context window",
    "too many tokens",
    "token limit",
)


class _DummyBuilder:
    def pack(self, state, questions, num_pause_tokens=None) -> PackedSequence:
        token_count = len(str(state).split())
        input_ids = torch.ones((1, token_count), dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return PackedSequence(
            text=str(state),
            input_ids=input_ids,
            attention_mask=attention_mask,
            questions={
                q_id: QuestionSpans(question_id=q_id, question_type="noul", query_token_idx=max(0, token_count - 1))
                for q_id in questions
            },
        )


class _DummyModel:
    def __init__(self, config=None, backbone=None) -> None:
        self.config = config
        self.backbone = backbone
        self.evaluate_calls = 0

    def evaluate_packed(self, packed: PackedSequence) -> dict[str, NoulResult]:
        self.evaluate_calls += 1
        return {q_id: NoulResult(noul=0.5) for q_id in packed.questions}


def _payload(token_count: int) -> dict:
    return {
        "model": "classone-qwen3.5-9b",
        "state": " ".join(["w"] * token_count),
        "questions": {"q": {"type": "noul", "instructions": "check?"}},
    }


def _install_dummy_pipeline(monkeypatch, model: _DummyModel) -> None:
    builder = _DummyBuilder()
    monkeypatch.setattr(server_app, "_model", model)
    monkeypatch.setattr(server_app, "_prompt_builder", builder)
    monkeypatch.setattr(server_app, "get_pipeline", lambda: (model, builder))


def test_default_limit_is_derived_from_model_config(monkeypatch) -> None:
    monkeypatch.delenv("CLASSONE_MAX_TOKENS_PER_REQUEST", raising=False)
    model = _DummyModel(config=SimpleNamespace(max_position_embeddings=2048))
    _install_dummy_pipeline(monkeypatch, model)
    assert server_app.get_max_tokens_per_request() == 2048


def test_default_limit_reads_nested_text_config(monkeypatch) -> None:
    monkeypatch.delenv("CLASSONE_MAX_TOKENS_PER_REQUEST", raising=False)
    model = _DummyModel(
        config=SimpleNamespace(),
        backbone=SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=131072))),
    )
    _install_dummy_pipeline(monkeypatch, model)
    assert server_app.get_max_tokens_per_request() == 131072


def test_20k_token_prompt_is_accepted_by_default(monkeypatch) -> None:
    monkeypatch.delenv("CLASSONE_MAX_TOKENS_PER_REQUEST", raising=False)
    model = _DummyModel(config=SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=131072)))
    _install_dummy_pipeline(monkeypatch, model)

    client = TestClient(server_app.app)
    response = client.post("/v1/decide", json=_payload(20000))
    assert response.status_code == 200, response.text
    assert response.json()["usage"]["input_tokens"] == 20000
    assert model.evaluate_calls == 1


@pytest.mark.parametrize(
    ("case", "token_count", "expected_ok"),
    [
        ("maximum_minus_one", 63, True),
        ("maximum", 64, True),
        ("maximum_plus_one", 65, False),
    ],
)
def test_prompt_boundaries_use_the_model_maximum(monkeypatch, case: str, token_count: int, expected_ok: bool) -> None:
    monkeypatch.delenv("CLASSONE_MAX_TOKENS_PER_REQUEST", raising=False)
    model = _DummyModel(config=SimpleNamespace(max_position_embeddings=64))
    _install_dummy_pipeline(monkeypatch, model)

    client = TestClient(server_app.app)
    response = client.post("/v1/decide", json=_payload(token_count))
    if expected_ok:
        assert response.status_code == 200, response.text
        assert response.json()["usage"]["input_tokens"] == token_count
    else:
        assert response.status_code in (400, 413, 422), response.text
        detail = response.json()["detail"]
        assert "64" in detail
        assert "8192" not in detail
        assert any(marker in detail.lower() for marker in CAPACITY_MARKERS)


def test_env_override_takes_precedence_over_model_config(monkeypatch) -> None:
    monkeypatch.setenv("CLASSONE_MAX_TOKENS_PER_REQUEST", "32")
    model = _DummyModel(config=SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=131072)))
    _install_dummy_pipeline(monkeypatch, model)

    client = TestClient(server_app.app)
    response = client.post("/v1/decide", json=_payload(33))
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "32" in detail
    assert any(marker in detail.lower() for marker in CAPACITY_MARKERS)
    assert model.evaluate_calls == 0


def test_over_limit_is_not_silently_truncated(monkeypatch) -> None:
    monkeypatch.delenv("CLASSONE_MAX_TOKENS_PER_REQUEST", raising=False)
    model = _DummyModel(config=SimpleNamespace(max_position_embeddings=20000))
    _install_dummy_pipeline(monkeypatch, model)

    client = TestClient(server_app.app)
    response = client.post("/v1/decide", json=_payload(20001))
    assert response.status_code == 400
    assert "20000" in response.text or "20,000" in response.text
    assert "20001" in response.text
    assert model.evaluate_calls == 0
