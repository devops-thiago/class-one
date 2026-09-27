"""Python Client SDK for ClassOne System 1 decision models.

Provides:
- Noul, Choice, Score question builder primitives
- Synchronous ClassOneClient
- Asynchronous AsyncClassOneClient
- Typed ClassOneResponse with grouped accessors (.nouls, .choices, .scores)
"""

from __future__ import annotations

import os
from typing import Any, Union

import httpx
from pydantic import BaseModel, Field

from classone.schemas import DecisionUsage


class Noul:
    """Boolean yes/no question builder primitive."""

    def __init__(self, instructions: str):
        self.instructions = instructions

    def to_dict(self) -> dict[str, Any]:
        return {"type": "noul", "instructions": self.instructions}


class Choice:
    """Categorical classification question builder primitive."""

    def __init__(self, instructions: str, criteria: dict[str, str]):
        self.instructions = instructions
        self.criteria = criteria

        if len(criteria) < 2:
            raise ValueError("Choice question must have at least 2 options in criteria.")
        if len(criteria) > 255:
            raise ValueError("Choice question cannot exceed 255 options.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "choice",
            "instructions": self.instructions,
            "criteria": self.criteria,
        }


class Score:
    """Ordinal rubric question builder primitive."""

    def __init__(self, instructions: str, criteria: list[str]):
        self.instructions = instructions
        self.criteria = criteria

        if len(criteria) < 2:
            raise ValueError("Score question must have at least 2 rubric levels.")
        if len(criteria) > 10:
            raise ValueError("Score question cannot exceed 10 rubric levels.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "score",
            "instructions": self.instructions,
            "criteria": self.criteria,
        }


class NoulAnswer(BaseModel):
    type: str = "noul"
    noul: float = Field(..., ge=0.0, le=1.0)


class ChoiceAnswer(BaseModel):
    type: str = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float = Field(..., ge=0.0, le=1.0)


class ScoreAnswer(BaseModel):
    type: str = "score"
    score: float
    probabilities: dict[str, float]
    confidence: float = Field(..., ge=0.0, le=1.0)


QuestionPrimitive = Union[Noul, Choice, Score]


class ClassOneResponse:
    """Structured response container from a ClassOne decision request."""

    def __init__(
        self,
        model: str,
        answers: dict[str, Any],
        usage: dict[str, int] | None = None,
    ):
        self.model = model
        self.raw_answers = answers
        self.usage = DecisionUsage(**(usage or {}))

        self.nouls: dict[str, NoulAnswer] = {}
        self.choices: dict[str, ChoiceAnswer] = {}
        self.scores: dict[str, ScoreAnswer] = {}
        self.answers: dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer] = {}

        for q_id, item in answers.items():
            if hasattr(item, "model_dump"):
                val = item.model_dump()
            elif isinstance(item, dict):
                val = item
            else:
                val = {"type": getattr(item, "type", None)}

            ans_type = val.get("type")
            if ans_type == "noul":
                noul_ans = NoulAnswer(noul=val["noul"])
                self.nouls[q_id] = noul_ans
                self.answers[q_id] = noul_ans
            elif ans_type == "choice":
                choice_ans = ChoiceAnswer(
                    choice=val["choice"],
                    probabilities=val["probabilities"],
                    confidence=val["confidence"],
                )
                self.choices[q_id] = choice_ans
                self.answers[q_id] = choice_ans
            elif ans_type == "score":
                score_ans = ScoreAnswer(
                    score=val["score"],
                    probabilities=val["probabilities"],
                    confidence=val["confidence"],
                )
                self.scores[q_id] = score_ans
                self.answers[q_id] = score_ans

    def __getitem__(self, key: str) -> NoulAnswer | ChoiceAnswer | ScoreAnswer:
        return self.answers[key]

    def __contains__(self, key: str) -> bool:
        return key in self.answers


class BaseClassOneClient:
    """Shared client configuration and payload formatting."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
    ):
        self.api_key = api_key or os.environ.get("CLASSONE_API_KEY", "default-key")
        self.base_url = (base_url or os.environ.get("CLASSONE_BASE_URL") or "http://localhost:8000").rstrip("/")
        self.timeout = timeout

    def _format_payload(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: dict[str, QuestionPrimitive],
        model: str,
    ) -> dict[str, Any]:
        serialized_questions = {}
        for q_id, q in questions.items():
            if hasattr(q, "to_dict"):
                serialized_questions[q_id] = q.to_dict()
            elif isinstance(q, dict):
                serialized_questions[q_id] = q
            else:
                raise TypeError(f"Invalid question object for '{q_id}': {type(q)}")

        return {
            "model": model,
            "state": state,
            "questions": serialized_questions,
        }

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


class ClassOneClient(BaseClassOneClient):
    """Synchronous client for ClassOne decision models."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        http_client: Any | None = None,
    ):
        super().__init__(api_key=api_key, base_url=base_url, timeout=timeout)
        if http_client is not None:
            self._client = http_client
            self._owns_client = False
        else:
            self._client = httpx.Client(
                base_url=self.base_url,
                headers=self._headers(),
                timeout=self.timeout,
                transport=transport,
            )
            self._owns_client = True

    def __enter__(self) -> ClassOneClient:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if getattr(self, "_owns_client", True):
            self._client.close()

    def decide(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: dict[str, QuestionPrimitive],
        model: str = "class-one-gemma-4-e2b-it",
    ) -> ClassOneResponse:
        """Evaluates questions over state in a single synchronous call."""
        payload = self._format_payload(state=state, questions=questions, model=model)
        resp = self._client.post("/v1/decide", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return ClassOneResponse(
            model=data.get("model", model),
            answers=data.get("answers", {}),
            usage=data.get("usage", {}),
        )


class AsyncClassOneClient(BaseClassOneClient):
    """Asynchronous client for ClassOne decision models."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
        http_client: httpx.AsyncClient | None = None,
    ):
        super().__init__(api_key=api_key, base_url=base_url, timeout=timeout)
        if http_client is not None:
            self._client = http_client
            self._owns_client = False
        else:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._headers(),
                timeout=self.timeout,
                transport=transport,
            )
            self._owns_client = True

    async def __aenter__(self) -> AsyncClassOneClient:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        if getattr(self, "_owns_client", True):
            await self._client.aclose()

    async def decide(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: dict[str, QuestionPrimitive],
        model: str = "class-one-gemma-4-e2b-it",
    ) -> ClassOneResponse:
        """Evaluates questions over state in an asynchronous call."""
        payload = self._format_payload(state=state, questions=questions, model=model)
        resp = await self._client.post("/v1/decide", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return ClassOneResponse(
            model=data.get("model", model),
            answers=data.get("answers", {}),
            usage=data.get("usage", {}),
        )
