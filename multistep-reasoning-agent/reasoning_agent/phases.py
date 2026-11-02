"""The three phases of the agent: Planner, Executor and Verifier.

Each phase gets the LLM client injected (any object with `generate`), builds its prompt from
prompts.py and parses the reply into a Pydantic model from schemas.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from . import prompts
from .llm import LLMClient
from .schemas import Check, Execution, IndependentSolution, Plan, Review, parse_llm_json
from .tools import CalculationError, evaluate, numbers_match


class Planner:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def plan(self, question: str, feedback: Optional[List[str]] = None) -> Plan:
        reply = self.llm.generate(prompts.PLANNER_SYSTEM, prompts.planner_user(question, feedback), json_mode=True)
        return parse_llm_json(reply, Plan)


class Executor:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def execute(self, question: str, plan: Plan, feedback: Optional[List[str]] = None) -> Execution:
        reply = self.llm.generate(prompts.EXECUTOR_SYSTEM,
                                  prompts.executor_user(question, plan.steps, feedback), json_mode=True)
        return parse_llm_json(reply, Execution)


@dataclass
class Verification:
    passed: bool
    checks: List[Check]
    issues: List[str]
    independent: Optional[IndependentSolution] = None
    review: Optional[Review] = None
    extra: dict = field(default_factory=dict)


class Verifier:
    """Checks a proposed solution three ways:

    1. arithmetic_recomputed (code): every calculation is re-computed with the safe calculator.
    2. independent_resolve_match (LLM + code): a fresh LLM call solves the question without
       seeing the proposed solution; numeric answers are compared in code.
    3. review (LLM): compares both answers, validates constraints and looks for inconsistencies.
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def verify(self, question: str, execution: Execution) -> Verification:
        checks = [check_arithmetic(execution)]

        independent = parse_llm_json(
            self.llm.generate(prompts.RESOLVER_SYSTEM, prompts.resolver_user(question), json_mode=True),
            IndependentSolution)
        checks.append(check_independent_match(execution, independent))

        proposed = execution.model_dump(exclude={"numeric_answer", "answer_unit"})
        review = parse_llm_json(
            self.llm.generate(prompts.REVIEWER_SYSTEM,
                              prompts.reviewer_user(question, proposed, independent.model_dump()), json_mode=True),
            Review)
        checks.extend(review.checks)
        checks.append(Check(check_name="reviewer_verdict", passed=review.verdict == "approve",
                            details="approved" if review.verdict == "approve" else "rejected"))

        issues = [f"{c.check_name}: {c.details}" for c in checks[:2] if not c.passed] + review.issues
        if not issues and not all(c.passed for c in checks):
            issues = [f"{c.check_name}: {c.details}" for c in checks if not c.passed]
        return Verification(passed=all(c.passed for c in checks), checks=checks, issues=issues,
                            independent=independent, review=review)


def check_arithmetic(execution: Execution) -> Check:
    """Re-compute every calculation the executor wrote and compare with its claimed result."""
    problems, count = [], 0
    for step in execution.steps:
        if not step.calculation:
            continue
        count += 1
        try:
            value = evaluate(step.calculation)
        except CalculationError as err:
            problems.append(f"step {step.step}: {err}")
            continue
        try:
            claimed = float(step.result)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            problems.append(f"step {step.step}: result {step.result!r} is not a number")
            continue
        if not numbers_match(value, claimed):
            problems.append(f"step {step.step}: {step.calculation} = {value:g}, not {claimed:g}")
    if problems:
        return Check(check_name="arithmetic_recomputed", passed=False, details="; ".join(problems))
    details = (f"re-computed {count} calculation(s) in Python; all match" if count
               else "no calculations to re-compute")
    return Check(check_name="arithmetic_recomputed", passed=True, details=details)


def _unit(unit: Optional[str]) -> str:
    return (unit or "").strip().lower().rstrip("s")


def check_independent_match(execution: Execution, independent: IndependentSolution) -> Check:
    """Compare the executor's answer with the independent re-solve, in code where possible."""
    name = "independent_resolve_match"
    if execution.solvable != independent.solvable:
        return Check(check_name=name, passed=False,
                     details=f"proposed solution says solvable={execution.solvable}, "
                             f"independent re-solve says solvable={independent.solvable}")
    a, b = execution.numeric_answer, independent.numeric_answer
    if a is None or b is None:
        return Check(check_name=name, passed=True,
                     details=f"answer is not a single number; compared by the reviewer "
                             f"(independent answer: {independent.final_answer!r})")
    if _unit(execution.answer_unit) != _unit(independent.answer_unit):
        return Check(check_name=name, passed=True,
                     details=f"units differ ({execution.answer_unit!r} vs {independent.answer_unit!r}); "
                             f"compared by the reviewer")
    if numbers_match(a, b):
        return Check(check_name=name, passed=True,
                     details=f"independent re-solve agrees: {b:g} {independent.answer_unit or ''}".strip())
    return Check(check_name=name, passed=False,
                 details=f"proposed {a:g} vs independent {b:g} {independent.answer_unit or ''}".strip())
