"""Sanity checks for the evaluation set and its grading patterns (no LLM calls)."""
import json
from pathlib import Path

import pytest

from evals.run_evals import grade

CASES = json.loads((Path(__file__).resolve().parent.parent / "evals/questions.json").read_text())


def fake_response(answer, status="success"):
    return {"answer": answer, "status": status, "reasoning_visible_to_user": ""}


def test_suite_has_enough_easy_and_tricky_questions():
    categories = [c["category"] for c in CASES]
    assert 5 <= categories.count("easy") <= 10
    assert 3 <= categories.count("tricky") <= 7
    assert len({c["id"] for c in CASES}) == len(CASES)


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_expected_answer_is_graded_correct(case):
    assert grade(case, fake_response(case["expected"], case.get("expected_status", "success")))


def test_grading_rejects_wrong_answers():
    by_id = {c["id"]: c for c in CASES}
    assert not grade(by_id["E1"], fake_response("4 hours 35 minutes"))
    assert not grade(by_id["E3"], fake_response("09:45–10:30 and 11:00–12:00"))
    assert not grade(by_id["T2"], fake_response("$0.10"))
    assert not grade(by_id["T6"], fake_response("-2 apples"))
    assert not grade(by_id["E2"], fake_response("9 apples", status="failed"))
