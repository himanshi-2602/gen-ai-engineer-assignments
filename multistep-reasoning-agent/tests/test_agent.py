"""Agent loop tests with a MockLLM: each test scripts exactly what the 'model' replies."""
import json

from helpers import (QUESTION, execution_reply, good_attempt, independent_reply, plan_reply,
                     review_reply)
from reasoning_agent.agent import ReasoningAgent
from reasoning_agent.llm import LLMError, MockLLM
from reasoning_agent.phases import check_arithmetic, check_independent_match
from reasoning_agent.schemas import Execution, IndependentSolution
from reasoning_agent.prompts import EXECUTOR_SYSTEM, PLANNER_SYSTEM, RESOLVER_SYSTEM, REVIEWER_SYSTEM


def make_agent(replies, max_retries=2, tmp_path=None):
    llm = MockLLM(replies)
    trace = str(tmp_path / "trace.jsonl") if tmp_path else None
    return ReasoningAgent(llm, max_retries=max_retries, trace_path=trace), llm


def test_success_on_first_attempt(tmp_path):
    agent, llm = make_agent(good_attempt(), tmp_path=tmp_path)
    response = agent.solve(QUESTION)

    assert response.status == "success"
    assert response.answer == "3 hours 35 minutes"
    assert response.metadata.retries == 0
    assert all(c.passed for c in response.metadata.checks)
    names = [c.check_name for c in response.metadata.checks]
    assert names[:2] == ["arithmetic_recomputed", "independent_resolve_match"]
    assert len(llm.calls) == 4  # planner, executor, independent re-solve, reviewer
    assert all(call["json_mode"] for call in llm.calls)


def test_response_matches_required_schema():
    agent, _ = make_agent(good_attempt())
    data = agent.solve(QUESTION).model_dump()
    assert set(data) == {"answer", "status", "reasoning_visible_to_user", "metadata"}
    assert set(data["metadata"]) == {"plan", "checks", "retries"}
    assert set(data["metadata"]["checks"][0]) == {"check_name", "passed", "details"}


def test_chain_of_thought_is_not_in_the_response_but_is_logged(tmp_path):
    agent, _ = make_agent(good_attempt(), tmp_path=tmp_path)
    data = json.dumps(agent.solve(QUESTION).model_dump())
    assert "to_minutes" not in data  # intermediate calculations stay internal

    trace = json.loads((tmp_path / "trace.jsonl").read_text().splitlines()[0])
    assert trace["question"] == QUESTION
    assert "to_minutes" in json.dumps(trace["attempts"][0]["execution"])
    assert trace["verifier_passed"] is True and trace["retries"] == 0


def test_retries_with_feedback_after_reviewer_rejects():
    first = [plan_reply(), execution_reply(final="4 hours 35 minutes", numeric=275, result=275,
                                           calculation="275"),
             independent_reply(), review_reply(approve=False, issues=["Wrong duration: should be 215 minutes"])]
    agent, llm = make_agent(first + good_attempt())
    response = agent.solve(QUESTION)

    assert response.status == "success"
    assert response.metadata.retries == 1
    assert any(not c.passed and "[attempt 1]" in c.check_name for c in response.metadata.checks)
    # The second planner call includes the verifier's feedback.
    assert "Wrong duration" in llm.calls[4]["user"]


def test_fails_after_max_retries():
    bad_attempt = [plan_reply(), execution_reply(), independent_reply(numeric=300), review_reply(approve=False,
                                                                                                issues=["disagree"])]
    agent, llm = make_agent(bad_attempt * 3, max_retries=2)
    response = agent.solve(QUESTION)

    assert response.status == "failed"
    assert response.answer == "Unable to produce a verified answer."
    assert response.metadata.retries == 2
    assert "3 attempt(s)" in response.reasoning_visible_to_user
    assert len(llm.calls) == 12


def test_invalid_json_counts_as_failed_attempt_and_retries():
    agent, _ = make_agent(["this is not json"] + good_attempt())
    response = agent.solve(QUESTION)
    assert response.status == "success"
    assert response.metadata.retries == 1
    assert response.metadata.checks[0].check_name == "output_format [attempt 1]"


def test_wrong_arithmetic_is_caught_by_code_even_if_reviewer_approves():
    bad = [plan_reply(), execution_reply(result=225, numeric=215), independent_reply(), review_reply()]
    agent, _ = make_agent(bad + good_attempt())
    response = agent.solve(QUESTION)
    failed = [c for c in response.metadata.checks if not c.passed]
    assert failed[0].check_name == "arithmetic_recomputed [attempt 1]"
    assert "= 215, not 225" in failed[0].details
    assert response.metadata.retries == 1


def test_verified_impossible_question_is_reported_as_failed_with_reason():
    replies = [plan_reply(),
               execution_reply(final="Not possible: she cannot give away 5 apples when she has 3.",
                               numeric=None, solvable=False, calculation="3 - 5", result=-2),
               independent_reply(final="Impossible", numeric=None, solvable=False),
               review_reply()]
    agent, _ = make_agent(replies)
    response = agent.solve("Alice has 3 apples and gives away 5. How many are left?")
    assert response.status == "failed"
    assert response.answer.startswith("Not possible")
    assert response.metadata.retries == 0


def test_llm_outage_returns_failed_response_instead_of_crashing():
    agent, _ = make_agent([])  # MockLLM raises LLMError immediately
    response = agent.solve(QUESTION)
    assert response.status == "failed"
    assert "unavailable" in response.answer


def test_check_independent_match():
    execution = Execution.model_validate(json.loads(execution_reply()))
    agree = IndependentSolution.model_validate(json.loads(independent_reply()))
    disagree = IndependentSolution.model_validate(json.loads(independent_reply(numeric=200)))
    not_numeric = IndependentSolution.model_validate(json.loads(independent_reply(numeric=None)))
    assert check_independent_match(execution, agree).passed
    assert not check_independent_match(execution, disagree).passed
    assert check_independent_match(execution, not_numeric).passed  # left to the reviewer


def test_check_arithmetic_reports_unsupported_function():
    execution = Execution.model_validate(json.loads(execution_reply(calculation="sqrt(16)", result=4)))
    check = check_arithmetic(execution)
    assert not check.passed and "unsupported function 'sqrt'" in check.details


def test_prompts_define_output_format_and_examples():
    for prompt in (PLANNER_SYSTEM, EXECUTOR_SYSTEM, RESOLVER_SYSTEM, REVIEWER_SYSTEM):
        assert "Output JSON only" in prompt
        assert prompt.count("Example") >= 2


def test_solve_function_returns_plain_dict(monkeypatch):
    import reasoning_agent.agent as agent_module
    from reasoning_agent import solve

    monkeypatch.setattr(agent_module, "_default_agent", ReasoningAgent(MockLLM(good_attempt())))
    result = solve(QUESTION)
    assert isinstance(result, dict) and result["answer"] == "3 hours 35 minutes"
