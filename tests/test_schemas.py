"""Unit tests for ClassOne System 1 decision schemas."""

import pytest
from pydantic import ValidationError

from classone.schemas import (
    ChoiceQuestion,
    ClassOneRequest,
    DecisionRequest,
    NoulQuestion,
    NoulResult,
    ScoreQuestion,
)


def test_noul_question_and_result():
    q = NoulQuestion(instructions="Is this request about billing?")
    assert q.type == "noul"

    res = NoulResult(noul=0.87)
    assert res.noul == 0.87

    with pytest.raises(ValidationError):
        NoulResult(noul=1.5)


def test_choice_question_validation():
    # Valid choice
    q = ChoiceQuestion(
        instructions="Select target department",
        criteria={"sales": "Sales and pricing", "tech": "Technical support"},
    )
    assert len(q.criteria) == 2

    # Less than 2 options should fail
    with pytest.raises(ValidationError):
        ChoiceQuestion(
            instructions="Invalid choice",
            criteria={"sales": "Sales only"},
        )


def test_score_question_validation():
    # Valid score rubric
    q = ScoreQuestion(
        instructions="Rate ticket severity",
        criteria=["low", "medium", "high", "critical"],
    )
    assert len(q.criteria) == 4

    # Less than 2 levels should fail
    with pytest.raises(ValidationError):
        ScoreQuestion(instructions="Bad score", criteria=["only_one"])

    # More than 10 levels should fail
    with pytest.raises(ValidationError):
        ScoreQuestion(instructions="Bad score", criteria=[f"level_{i}" for i in range(11)])


def test_decision_request_roundtrip():
    req = DecisionRequest(
        state={"message": "I was double charged"},
        questions={
            "refund": NoulQuestion(instructions="Is customer asking for a refund?"),
            "dept": ChoiceQuestion(
                instructions="Which department?",
                criteria={"billing": "Payment issues", "support": "General help"},
            ),
            "urgency": ScoreQuestion(
                instructions="How urgent?",
                criteria=["normal", "urgent", "emergency"],
            ),
        },
    )
    dumped = req.model_dump()
    assert dumped["questions"]["refund"]["type"] == "noul"
    assert dumped["questions"]["dept"]["type"] == "choice"
    assert dumped["questions"]["urgency"]["type"] == "score"

    # Alias check
    alias_req = ClassOneRequest(state="test", questions={"q": NoulQuestion(instructions="test?")})
    assert alias_req.model == "class-one-gemma-4-e2b-it"
