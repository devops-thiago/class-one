"""Unit and integration tests for ClassOne FastAPI server."""

import pytest
from fastapi.testclient import TestClient

from classone.server.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_decide_endpoint(client):
    payload = {
        "model": "class-one-gemma-4-e2b-it",
        "state": {
            "customer": "Jordan",
            "message": "I was charged twice for my order #9948.",
        },
        "questions": {
            "refund": {
                "type": "noul",
                "instructions": "Is customer asking for a refund?",
            },
            "team": {
                "type": "choice",
                "instructions": "Which team should address this?",
                "criteria": {
                    "billing": "Charges and payouts",
                    "technical": "App bugs",
                },
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this?",
                "criteria": ["low", "normal", "high"],
            },
        },
    }

    response = client.post("/v1/decide", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["model"] == "class-one-gemma-4-e2b-it"
    assert "answers" in data

    # Check Noul answer
    assert "refund" in data["answers"]
    assert data["answers"]["refund"]["type"] == "noul"
    assert 0.0 <= data["answers"]["refund"]["noul"] <= 1.0

    # Check Choice answer
    assert "team" in data["answers"]
    assert data["answers"]["team"]["type"] == "choice"
    assert data["answers"]["team"]["choice"] in ["billing", "technical"]
    assert "billing" in data["answers"]["team"]["probabilities"]

    # Check Score answer
    assert "urgency" in data["answers"]
    assert data["answers"]["urgency"]["type"] == "score"
    assert 1.0 <= data["answers"]["urgency"]["score"] <= 3.0

    # Usage
    assert data["usage"]["input_tokens"] > 0
    assert data["usage"]["output_tokens"] == 0


def test_classone_endpoint_alias(client):
    payload = {
        "model": "class-one-gemma-4-e2b-it",
        "state": "User requested account deletion.",
        "questions": {
            "deletion": {
                "type": "noul",
                "instructions": "Is user requesting account deletion?",
            }
        },
    }
    response = client.post("/v1/classone", json=payload)
    assert response.status_code == 200
    assert "deletion" in response.json()["answers"]


def test_systemone_endpoint_alias(client):
    payload = {
        "model": "class-one-gemma-4-e2b-it",
        "state": "User requested account deletion.",
        "questions": {
            "deletion": {
                "type": "noul",
                "instructions": "Is user requesting account deletion?",
            }
        },
    }
    response = client.post("/v1/systemone", json=payload)
    assert response.status_code == 200
    assert "deletion" in response.json()["answers"]


def test_decide_max_questions_limit(client):
    too_many_questions = {
        f"q_{i}": {
            "type": "noul",
            "instructions": f"Question {i}",
        }
        for i in range(150)
    }
    payload = {
        "model": "class-one-gemma-4-e2b-it",
        "state": "Sample state",
        "questions": too_many_questions,
    }
    response = client.post("/v1/decide", json=payload)
    assert response.status_code == 400
    assert "Too many questions" in response.json()["detail"]
