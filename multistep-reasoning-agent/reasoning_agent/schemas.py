"""Data shapes, validated with Pydantic.

LLM outputs are parsed into these models, so a malformed reply (missing field, wrong type)
is caught immediately instead of breaking something later. The API request/response
models live here too.
"""
from __future__ import annotations

import json
import re
from typing import List, Literal, Optional, Type, TypeVar, Union

from pydantic import BaseModel, Field, ValidationError

# ---------- LLM outputs ----------


class Plan(BaseModel):
    steps: List[str] = Field(min_length=1, max_length=10)

    def as_text(self) -> str:
        return " ".join(f"{i}. {step}" for i, step in enumerate(self.steps, 1))


class ExecutionStep(BaseModel):
    step: int
    description: str
    calculation: Optional[str] = None  # arithmetic expression, re-computed by tools.evaluate
    result: Optional[Union[float, str]] = None


class Execution(BaseModel):
    steps: List[ExecutionStep] = Field(min_length=1)
    solvable: bool = True
    final_answer: str
    numeric_answer: Optional[float] = None
    answer_unit: Optional[str] = None
    explanation: str


class IndependentSolution(BaseModel):
    work: str
    solvable: bool = True
    final_answer: str
    numeric_answer: Optional[float] = None
    answer_unit: Optional[str] = None


class Check(BaseModel):
    check_name: str
    passed: bool
    details: str


class Review(BaseModel):
    checks: List[Check]
    verdict: Literal["approve", "reject"]
    issues: List[str] = []


# ---------- API ----------


class SolveRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000, examples=[
        "If a train leaves at 14:30 and arrives at 18:05, how long is the journey?"])


class Metadata(BaseModel):
    plan: str
    checks: List[Check]
    retries: int


class SolveResponse(BaseModel):
    answer: str
    status: Literal["success", "failed"]
    reasoning_visible_to_user: str
    metadata: Metadata


# ---------- parsing ----------

class OutputFormatError(ValueError):
    """The LLM reply was not valid JSON in the expected shape."""


Model = TypeVar("Model", bound=BaseModel)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def parse_llm_json(text: str, model: Type[Model]) -> Model:
    """Parse an LLM reply into `model`, tolerating ```json fences around the JSON."""
    cleaned = _FENCE.sub("", text.strip())
    try:
        return model.model_validate(json.loads(cleaned))
    except (json.JSONDecodeError, ValidationError) as err:
        raise OutputFormatError(f"{model.__name__} output was invalid: {str(err)[:300]}") from err
