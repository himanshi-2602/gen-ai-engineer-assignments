"""The agent loop: plan -> execute -> verify, retrying with feedback when verification fails.

Only `answer` and `reasoning_visible_to_user` are meant for the user. The plan, checks and
retry count go in `metadata`; the full internal trace (every step, intermediate value and
verifier output) is written to a log file, never into the response.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from .config import Settings
from .llm import LLMClient, LLMError, make_llm
from .phases import Executor, Planner, Verifier
from .schemas import Check, Metadata, OutputFormatError, SolveResponse


class ReasoningAgent:
    def __init__(self, llm: LLMClient, max_retries: int = 2, trace_path: Optional[str] = None):
        self.planner = Planner(llm)
        self.executor = Executor(llm)
        self.verifier = Verifier(llm)
        self.max_retries = max_retries
        self.trace_path = Path(trace_path) if trace_path else None

    def solve(self, question: str) -> SolveResponse:
        return self.solve_with_trace(question)[0]

    def solve_with_trace(self, question: str) -> Tuple[SolveResponse, dict]:
        started = time.monotonic()
        trace: dict = {"question": question, "attempts": []}
        checks_by_attempt: List[Tuple[int, List[Check]]] = []
        feedback: Optional[List[str]] = None
        plan_text, last_answer, last_issues = "", None, []
        response = None

        for attempt in range(self.max_retries + 1):
            record: dict = {"attempt": attempt + 1}
            trace["attempts"].append(record)
            try:
                plan = self.planner.plan(question, feedback)
                plan_text = plan.as_text()
                record["plan"] = plan.steps
                execution = self.executor.execute(question, plan, feedback)
                record["execution"] = execution.model_dump()
                last_answer = execution.final_answer
                verification = self.verifier.verify(question, execution)
            except OutputFormatError as err:
                # The model broke the JSON contract: count it as a failed attempt and retry.
                record["error"] = str(err)
                checks_by_attempt.append((attempt, [Check(check_name="output_format", passed=False,
                                                          details=str(err))]))
                last_issues = feedback = [f"Your previous reply was not valid JSON in the required format: {err}"]
                continue
            except LLMError as err:
                record["error"] = str(err)
                response = self._response(
                    answer="Sorry, the reasoning service is unavailable right now. Please try again later.",
                    status="failed", reasoning=f"The language model could not be reached ({err}).",
                    plan=plan_text, checks_by_attempt=checks_by_attempt, retries=attempt)
                break

            record["verification"] = {
                "passed": verification.passed,
                "checks": [c.model_dump() for c in verification.checks],
                "issues": verification.issues,
                "independent": verification.independent.model_dump() if verification.independent else None,
            }
            checks_by_attempt.append((attempt, verification.checks))

            if verification.passed:
                response = self._response(
                    answer=execution.final_answer,
                    # A verified "this question has no valid answer" is reported as failed, with the reason.
                    status="success" if execution.solvable else "failed",
                    reasoning=execution.explanation,
                    plan=plan_text, checks_by_attempt=checks_by_attempt, retries=attempt)
                break
            last_issues = feedback = verification.issues

        if response is None:  # every attempt failed verification
            attempts = self.max_retries + 1
            last = f" The last proposed answer was {last_answer!r}, but it did not pass the checks." if last_answer else ""
            response = self._response(
                answer="Unable to produce a verified answer.",
                status="failed",
                reasoning=f"Verification failed after {attempts} attempt(s).{last} "
                          f"Problems found: {'; '.join(last_issues)[:500]}",
                plan=plan_text, checks_by_attempt=checks_by_attempt, retries=self.max_retries)

        final_checks = checks_by_attempt[-1][1] if checks_by_attempt else []
        trace.update({
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seconds": round(time.monotonic() - started, 2),
            "verifier_passed": bool(final_checks) and all(c.passed for c in final_checks),
            "retries": response.metadata.retries,
            "response": response.model_dump(),
        })
        self._write_trace(trace)
        return response, trace

    @staticmethod
    def _response(answer: str, status: str, reasoning: str, plan: str,
                  checks_by_attempt: List[Tuple[int, List[Check]]], retries: int) -> SolveResponse:
        multiple = len(checks_by_attempt) > 1
        checks = [
            Check(check_name=f"{c.check_name} [attempt {attempt + 1}]" if multiple else c.check_name,
                  passed=c.passed, details=c.details)
            for attempt, attempt_checks in checks_by_attempt for c in attempt_checks
        ]
        return SolveResponse(answer=answer, status=status, reasoning_visible_to_user=reasoning,
                             metadata=Metadata(plan=plan, checks=checks, retries=retries))

    def _write_trace(self, trace: dict) -> None:
        if not self.trace_path:
            return
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")


def build_agent(settings: Optional[Settings] = None) -> ReasoningAgent:
    """Create an agent from environment settings (used by the CLI, API and eval script)."""
    settings = settings or Settings.from_env()
    return ReasoningAgent(make_llm(settings), settings.max_retries, settings.trace_path)


_default_agent: Optional[ReasoningAgent] = None


def solve(question: str) -> dict:
    """Notebook-style entry point: solve(question) -> dict with the required JSON schema."""
    global _default_agent
    if _default_agent is None:
        _default_agent = build_agent()
    return _default_agent.solve(question).model_dump()
