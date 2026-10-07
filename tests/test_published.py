"""Verification tests for the published Hugging Face model and PyPI package."""

import pytest
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from classone import Choice, ClassOneClient, Noul, Score
from classone.modeling.modeling_classone import ClassOneModel
from classone.schemas import ChoiceQuestion, DecisionRequest, NoulQuestion, ScoreQuestion
from classone.tokenizer import SPECIAL_TOKENS, ClassOnePromptBuilder

HF_REPO_ID = "devops-thiago/classone-gemma4-e2b"


def test_pypi_package_exports():
    """Verifies that all primary SDK and schema primitives are exported cleanly."""
    assert ClassOneClient is not None
    assert Noul is not None
    assert Choice is not None
    assert Score is not None

    req = DecisionRequest(
        model="devops-thiago/classone-gemma4-e2b",
        state={"message": "System operational"},
        questions={"check": NoulQuestion(instructions="Is system OK?")},
    )
    assert req.model == HF_REPO_ID
    assert "check" in req.questions


def test_published_tokenizer_and_special_tokens():
    """Verifies that the published tokenizer loads from HF Hub and has ClassOne special tokens."""
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID)
    vocab = tokenizer.get_vocab()
    for tok in SPECIAL_TOKENS:
        assert tok in vocab, f"Missing special token {tok} in published tokenizer"


def test_published_heads_structure():
    """Verifies that published classone_heads.pt file contains all expected heads and configs."""
    heads_path = hf_hub_download(HF_REPO_ID, "classone_heads.pt")
    data = torch.load(heads_path, map_location="cpu")

    assert "noul_head" in data
    assert "choice_head" in data
    assert "score_head" in data
    assert "config" in data

    # Verify hidden size is 1536 (matching Gemma 4 E2B hidden size)
    assert data["config"]["hidden_size"] == 1536

    # Verify temperature scaling parameters are present
    assert "temperature_raw" in data["noul_head"]
    assert "temperature_raw" in data["choice_head"]
    assert "choice_evaluator.temperature_raw" in data["score_head"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required for full model evaluation")
def test_published_model_inference_cuda():
    """Verifies end-to-end single-pass decision execution on CUDA using published model."""
    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID)
    builder = ClassOnePromptBuilder(tokenizer)

    model = ClassOneModel.from_backbone(
        base_model_name_or_path=HF_REPO_ID,
        tokenizer=tokenizer,
        device=device,
        torch_dtype=torch.float16,
    )

    heads_path = hf_hub_download(HF_REPO_ID, "classone_heads.pt")
    heads = torch.load(heads_path, map_location=device)
    model.noul_head.load_state_dict(heads["noul_head"])
    model.choice_head.load_state_dict(heads["choice_head"])
    model.score_head.load_state_dict(heads["score_head"])
    model.eval()

    state = {
        "customer": "Alex",
        "order_id": "ORD-9872",
        "message": "I was charged twice on my credit card. Please issue a refund.",
    }
    questions = {
        "refund_request": NoulQuestion(instructions="Is the customer requesting a refund?"),
        "routing": ChoiceQuestion(
            instructions="Target department:",
            criteria={"billing": "Payment & invoice disputes", "tech": "App bugs & outages"},
        ),
        "urgency": ScoreQuestion(
            instructions="Ticket urgency:",
            criteria=["low", "medium", "critical"],
        ),
    }

    packed = builder.pack(state=state, questions=questions)

    with torch.no_grad():
        results = model.evaluate_packed(packed)

    # Validate Noul result
    noul_res = results["refund_request"]
    assert noul_res.type == "noul"
    assert 0.0 <= noul_res.noul <= 1.0

    # Validate Choice result
    choice_res = results["routing"]
    assert choice_res.type == "choice"
    assert choice_res.choice in ["billing", "tech"]
    assert set(choice_res.probabilities.keys()) == {"billing", "tech"}
    assert abs(sum(choice_res.probabilities.values()) - 1.0) < 1e-3

    # Validate Score result
    score_res = results["urgency"]
    assert score_res.type == "score"
    assert 1.0 <= score_res.score <= 3.0
    assert abs(sum(score_res.probabilities.values()) - 1.0) < 1e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required for full model evaluation")
def test_published_model_multi_domain_scenarios():
    """Verifies that published model handles security and feedback domains with valid bounds."""
    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID)
    builder = ClassOnePromptBuilder(tokenizer)

    model = ClassOneModel.from_backbone(
        base_model_name_or_path=HF_REPO_ID,
        tokenizer=tokenizer,
        device=device,
        torch_dtype=torch.float16,
    )
    heads_path = hf_hub_download(HF_REPO_ID, "classone_heads.pt")
    heads = torch.load(heads_path, map_location=device)
    model.noul_head.load_state_dict(heads["noul_head"])
    model.choice_head.load_state_dict(heads["choice_head"])
    model.score_head.load_state_dict(heads["score_head"])
    model.eval()

    # Domain 1: Security Incident
    sec_state = {
        "event_type": "auth_failure",
        "ip": "198.51.100.24",
        "attempts": 45,
        "detail": "Repeated failed SSH logins with root user over 60 seconds",
    }
    sec_questions = {
        "is_brute_force": NoulQuestion(instructions="Is this an automated brute force attack?"),
        "mitigation": ChoiceQuestion(
            instructions="Automated action:",
            criteria={"block_ip": "Add IP to drop table", "mfa_challenge": "Require MFA", "ignore": "Normal activity"},
        ),
        "threat_severity": ScoreQuestion(
            instructions="Threat severity rating:",
            criteria=["low", "elevated", "high", "critical"],
        ),
    }
    packed_sec = builder.pack(state=sec_state, questions=sec_questions)
    with torch.no_grad():
        sec_res = model.evaluate_packed(packed_sec)

    assert 0.0 <= sec_res["is_brute_force"].noul <= 1.0
    assert sec_res["mitigation"].choice in ["block_ip", "mfa_challenge", "ignore"]
    assert 1.0 <= sec_res["threat_severity"].score <= 4.0

    # Domain 2: Customer Sentiment & Satisfaction
    fb_state = {
        "user": "Taylor",
        "nps": 10,
        "review": "The single-pass latency is incredible! Our microservices responded in under 70ms.",
    }
    fb_questions = {
        "is_complaint": NoulQuestion(instructions="Is this a negative complaint?"),
        "sentiment": ChoiceQuestion(
            instructions="Classify overall sentiment:",
            criteria={
                "positive": "Delighted customer",
                "neutral": "Informative feedback",
                "negative": "Unhappy customer",
            },
        ),
        "satisfaction": ScoreQuestion(
            instructions="Customer satisfaction level:",
            criteria=["very_dissatisfied", "dissatisfied", "neutral", "satisfied", "very_satisfied"],
        ),
    }
    packed_fb = builder.pack(state=fb_state, questions=fb_questions)
    with torch.no_grad():
        fb_res = model.evaluate_packed(packed_fb)

    assert 0.0 <= fb_res["is_complaint"].noul <= 1.0
    assert fb_res["sentiment"].choice in ["positive", "neutral", "negative"]
    assert 1.0 <= fb_res["satisfaction"].score <= 5.0
