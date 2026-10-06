"""Unit tests for the ClassOne Python Client SDK."""

import httpx
import pytest

from classone import (
    AsyncClassOneClient,
    Choice,
    ClassOneClient,
    Noul,
    Score,
)
from classone.server.app import app


def test_sdk_primitives_validation():
    # Valid primitives
    n = Noul(instructions="Is this valid?")
    assert n.to_dict()["type"] == "noul"

    c = Choice(instructions="Pick one", criteria={"a": "Option A", "b": "Option B"})
    assert c.to_dict()["type"] == "choice"

    s = Score(instructions="Rate this", criteria=["bad", "good"])
    assert s.to_dict()["type"] == "score"

    # Invalid choice (fewer than 2 options)
    with pytest.raises(ValueError):
        Choice(instructions="Bad", criteria={"only": "one"})

    # Invalid score (fewer than 2 levels)
    with pytest.raises(ValueError):
        Score(instructions="Bad", criteria=["only_one"])


def test_sync_client_integration():
    from fastapi.testclient import TestClient as APITestClient

    test_client = APITestClient(app)
    with ClassOneClient(http_client=test_client) as client:
        state = {
            "customer": "Jordan",
            "subject": "Payout failed again",
            "body": "My payouts have failed three times. Bank says fine.",
        }
        questions = {
            "queue": Choice(
                instructions="Which team should handle this ticket?",
                criteria={
                    "payments": "Payout failures and payment processing",
                    "account": "Login and account access",
                    "other": "Something else",
                },
            ),
            "escalate": Noul(instructions="Does this message require urgent human attention?"),
            "urgency": Score(
                instructions="How urgent is this ticket?",
                criteria=["can wait", "this week", "today"],
            ),
        }

        response = client.decide(
            state=state,
            questions=questions,
            model="class-one-gemma-4-e2b-it",
        )

        assert response.model == "class-one-gemma-4-e2b-it"
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens == 0

        # Choice answer access
        queue_ans = response.choices["queue"]
        assert queue_ans.choice in ["payments", "account", "other"]
        assert 0.0 <= queue_ans.confidence <= 1.0
        assert len(queue_ans.probabilities) == 3

        # Noul answer access
        escalate_ans = response.nouls["escalate"]
        assert 0.0 <= escalate_ans.noul <= 1.0

        # Score answer access
        urgency_ans = response.scores["urgency"]
        assert 1.0 <= urgency_ans.score <= 3.0
        assert 0.0 <= urgency_ans.confidence <= 1.0
        assert len(urgency_ans.probabilities) == 3

        # Indexing access
        assert "queue" in response
        assert response["queue"].choice == queue_ans.choice


@pytest.mark.asyncio
async def test_async_client_integration():
    transport = httpx.ASGITransport(app=app)
    async with AsyncClassOneClient(base_url="http://testserver", transport=transport) as client:
        state = "I need immediate help resetting my password."
        questions = {
            "is_auth": Noul(instructions="Is this related to authentication/passwords?"),
            "priority": Score(
                instructions="Urgency level",
                criteria=["low", "urgent"],
            ),
        }

        response = await client.decide(
            state=state,
            questions=questions,
        )

        assert "is_auth" in response.nouls
        assert 0.0 <= response.nouls["is_auth"].noul <= 1.0
        assert "priority" in response.scores
        assert 1.0 <= response.scores["priority"].score <= 2.0


def test_sdk_ambiguity_and_margin_helpers():
    """Verifies that ChoiceAnswer and ScoreAnswer compute is_decisive, margin, and is_ambiguous correctly."""
    from classone.sdk import ChoiceAnswer, ScoreAnswer

    # 1. Decisive choice (high confidence, large margin)
    decisive_choice = ChoiceAnswer(
        choice="billing",
        probabilities={"billing": 0.95, "tech": 0.05},
        confidence=0.90,
        is_decisive=True,
        margin=0.90,
    )
    assert decisive_choice.is_decisive is True
    assert decisive_choice.is_ambiguous is False
    assert decisive_choice.margin == 0.90

    # 2. Ambiguous choice (coin-flip, confidence 0.0, margin 0.0)
    ambiguous_choice = ChoiceAnswer(
        choice="billing",
        probabilities={"billing": 0.50, "tech": 0.50},
        confidence=0.0,
        is_decisive=False,
        margin=0.0,
    )
    assert ambiguous_choice.is_decisive is False
    assert ambiguous_choice.is_ambiguous is True
    assert ambiguous_choice.margin == 0.0

    # 3. Decisive score
    decisive_score = ScoreAnswer(
        score=2.0,
        probabilities={"low": 0.1, "urgent": 0.9},
        confidence=0.80,
        is_decisive=True,
        margin=0.80,
    )
    assert decisive_score.is_decisive is True
    assert decisive_score.is_ambiguous is False
