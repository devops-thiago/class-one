"""Schema definitions for ClassOne System 1 decision models.

Defines:
- Noul: Boolean evaluation returning calibrated P(true) in [0.0, 1.0]
- Choice: Categorical selection over 2 to 255 options with calibrated probability distribution
- Score: Continuous ordinal rubric scoring over 2 to 10 levels returning weighted expected value
- DecisionRequest & DecisionResponse: API payload models
"""

from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, Field, field_validator


class NoulQuestion(BaseModel):
    """Boolean yes/no question evaluated to a calibrated probability P(true)."""

    type: Literal["noul"] = "noul"
    instructions: str = Field(..., description="Prompt/instruction for the boolean check.")


class ChoiceQuestion(BaseModel):
    """Categorical classification over 2 to 255 options."""

    type: Literal["choice"] = "choice"
    instructions: str = Field(..., description="Prompt/instruction for selecting an option.")
    criteria: dict[str, str] = Field(
        ...,
        description="Dictionary mapping choice identifier to criteria/definition.",
    )

    @field_validator("criteria")
    @classmethod
    def validate_criteria(cls, v: dict[str, str]) -> dict[str, str]:
        if len(v) < 2:
            raise ValueError("Choice question must have at least 2 options.")
        if len(v) > 255:
            raise ValueError("Choice question cannot exceed 255 options.")
        for key, desc in v.items():
            if not key or not desc:
                raise ValueError("Choice keys and descriptions must not be empty.")
        return v


class ScoreQuestion(BaseModel):
    """Ordinal evaluation along an ordered rubric of 2 to 10 levels."""

    type: Literal["score"] = "score"
    instructions: str = Field(..., description="Prompt/instruction for scoring.")
    criteria: list[str] = Field(
        ...,
        description="Ordered list of rubric levels from lowest to highest.",
    )

    @field_validator("criteria")
    @classmethod
    def validate_levels(cls, v: list[str]) -> list[str]:
        if len(v) < 2:
            raise ValueError("Score question must have at least 2 rubric levels.")
        if len(v) > 10:
            raise ValueError("Score question cannot exceed 10 rubric levels.")
        return v


Question = Union[NoulQuestion, ChoiceQuestion, ScoreQuestion]


class NoulResult(BaseModel):
    """Result of a Noul question."""

    type: Literal["noul"] = "noul"
    noul: float = Field(..., ge=0.0, le=1.0, description="Calibrated probability P(true).")


class ChoiceResult(BaseModel):
    """Result of a Choice question."""

    type: Literal["choice"] = "choice"
    choice: str = Field(..., description="Selected option key.")
    probabilities: dict[str, float] = Field(
        ...,
        description="Probability distribution across all available choices.",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Calibrated decision confidence.")


class ScoreResult(BaseModel):
    """Result of a Score question."""

    type: Literal["score"] = "score"
    score: float = Field(..., description="Probability-weighted fractional score.")
    probabilities: dict[str, float] = Field(
        ...,
        description="Probability distribution across each ordered rubric level.",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the score outcome.")


Result = Union[NoulResult, ChoiceResult, ScoreResult]


class DecisionUsage(BaseModel):
    """Token usage metrics."""

    input_tokens: int = 0
    output_tokens: int = 0  # In System 1 models this is 0 because there is no autoregressive decode


class DecisionRequest(BaseModel):
    """Request payload for ClassOne decision inference."""

    model: str = Field("class-one-gemma-4-e2b-it", description="Model name or pinned checkpoint.")
    state: str | dict[str, Any] | list[Any] = Field(
        ...,
        description="Unstructured or semi-structured state (text, JSON object, or list).",
    )
    questions: dict[str, Question] = Field(
        ...,
        description="Map of question key to typed Question definition.",
    )

    @field_validator("questions")
    @classmethod
    def validate_questions_non_empty(cls, v: dict[str, Question]) -> dict[str, Question]:
        if not v:
            raise ValueError("At least one question must be provided in the request.")
        return v


class DecisionResponse(BaseModel):
    """Response payload for ClassOne decision inference."""

    model: str
    answers: dict[str, Result]
    usage: DecisionUsage = Field(default_factory=DecisionUsage)


# Ergonomic Aliases
ClassOneUsage = DecisionUsage
ClassOneRequest = DecisionRequest
ClassOneResponse = DecisionResponse
