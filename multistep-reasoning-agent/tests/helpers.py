"""Builders for fake LLM replies, so tests read like the real JSON the model returns."""
import json

QUESTION = "If a train leaves at 14:30 and arrives at 18:05, how long is the journey?"


def plan_reply(*steps):
    return json.dumps({"steps": list(steps or ["Extract times", "Subtract", "Format"])})


def execution_reply(final="3 hours 35 minutes", numeric=215, result=215, solvable=True,
                    calculation="to_minutes('18:05') - to_minutes('14:30')"):
    return json.dumps({
        "steps": [{"step": 1, "description": "minutes between", "calculation": calculation, "result": result}],
        "solvable": solvable, "final_answer": final, "numeric_answer": numeric,
        "answer_unit": "minutes" if numeric is not None else None,
        "explanation": "From 14:30 to 18:05 is 215 minutes.",
    })


def independent_reply(final="3 hours 35 minutes", numeric=215, solvable=True):
    return json.dumps({"work": "18:05 - 14:30 = 215 min", "solvable": solvable, "final_answer": final,
                       "numeric_answer": numeric, "answer_unit": "minutes" if numeric is not None else None})


def review_reply(approve=True, issues=()):
    names = ["answers_agree", "constraints_valid", "explanation_consistent", "answers_question"]
    return json.dumps({
        "checks": [{"check_name": n, "passed": approve, "details": "ok" if approve else "problem"} for n in names],
        "verdict": "approve" if approve else "reject",
        "issues": list(issues),
    })


def good_attempt():
    """The 4 replies of one successful attempt: plan, execution, independent re-solve, review."""
    return [plan_reply(), execution_reply(), independent_reply(), review_reply()]
