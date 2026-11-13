"""Regression tests for GitHub issue #19: prompt-length HTTP 400 capacity marker & env limit."""

from __future__ import annotations

from fastapi.testclient import TestClient

from classone.server import app as server_app

CAPACITY_MARKERS = (
    "context length",
    "maximum context length",
    "context window",
    "too many tokens",
    "token limit",
)


def _payload(word_count: int) -> dict:
    return {
        "model": "class-one-gemma-4-e2b-it",
        "state": " ".join(["tok"] * word_count),
        "questions": {
            "q1": {
                "type": "noul",
                "instructions": "ok?",
            }
        },
    }


def _count_tokens(word_count: int) -> int:
    server_app.get_pipeline()
    assert server_app._prompt_builder is not None
    req = _payload(word_count)
    from classone.schemas import DecisionRequest

    parsed = DecisionRequest(**req)
    packed = server_app._prompt_builder.pack(state=parsed.state, questions=parsed.questions)
    return int(packed.input_ids.shape[1])


def _find_word_count_for_limit(target_tokens: int) -> int:
    # WordLevel whitespace tokenizer adds ~1 token per extra word in state
    low_words = 1
    base = _count_tokens(low_words)
    guess = max(1, low_words + (target_tokens - base))
    while _count_tokens(guess) != target_tokens:
        diff = target_tokens - _count_tokens(guess)
        guess = max(1, guess + diff)
    return guess


def test_standalone_without_config_window_is_not_capped_at_8192(monkeypatch) -> None:
    monkeypatch.delenv("CLASSONE_MAX_TOKENS_PER_REQUEST", raising=False)
    server_app._model = None
    assert server_app.get_max_tokens_per_request() is None


def test_prompt_below_and_at_limit_succeeds(monkeypatch) -> None:
    monkeypatch.setenv("CLASSONE_MAX_TOKENS_PER_REQUEST", "32")
    client = TestClient(server_app.app)

    words_at_32 = _find_word_count_for_limit(32)
    resp_below = client.post("/v1/decide", json=_payload(words_at_32 - 1))
    assert resp_below.status_code == 200
    assert resp_below.json()["usage"]["input_tokens"] == 31

    resp_at = client.post("/v1/decide", json=_payload(words_at_32))
    assert resp_at.status_code == 200
    assert resp_at.json()["usage"]["input_tokens"] == 32


def test_prompt_above_env_limit_returns_400_with_capacity_marker(monkeypatch) -> None:
    monkeypatch.setenv("CLASSONE_MAX_TOKENS_PER_REQUEST", "32")
    client = TestClient(server_app.app)

    words_at_33 = _find_word_count_for_limit(33)
    resp = client.post("/v1/decide", json=_payload(words_at_33))
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "33" in detail
    assert "32" in detail
    assert any(marker in detail.lower() for marker in CAPACITY_MARKERS)


def test_model_config_limit_rejects_excess_tokens_with_capacity_marker(monkeypatch) -> None:
    monkeypatch.delenv("CLASSONE_MAX_TOKENS_PER_REQUEST", raising=False)
    client = TestClient(server_app.app)
    model, builder = server_app.get_pipeline()
    monkeypatch.setattr(model.config, "max_position_embeddings", 40, raising=False)

    words_at_41 = _find_word_count_for_limit(41)
    resp = client.post("/v1/decide", json=_payload(words_at_41))
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "41" in detail
    assert "40" in detail
    assert any(marker in detail.lower() for marker in CAPACITY_MARKERS)


def test_env_override_applies_dynamically_without_reimport(monkeypatch) -> None:
    client = TestClient(server_app.app)
    words_at_40 = _find_word_count_for_limit(40)

    monkeypatch.setenv("CLASSONE_MAX_TOKENS_PER_REQUEST", "32")
    assert client.post("/v1/decide", json=_payload(words_at_40)).status_code == 400

    monkeypatch.setenv("CLASSONE_MAX_TOKENS_PER_REQUEST", "64")
    assert client.post("/v1/decide", json=_payload(words_at_40)).status_code == 200
