import pytest

from reasoning_agent.schemas import Execution, OutputFormatError, Plan, parse_llm_json
from reasoning_agent.tools import CalculationError, evaluate, numbers_match, to_minutes


@pytest.mark.parametrize("expression, expected", [
    ("3 + 2 * 3", 9),
    ("(5 - 2 + 12) / 3", 5),
    ("215 // 60", 3),
    ("215 % 60", 35),
    ("-4 + 10", 6),
    ("to_minutes('18:05') - to_minutes('14:30')", 215),
    ("to_minutes('01:15') + 1440 - to_minutes('22:40')", 155),
    ("max(30, 45, 60) - min(30, 45)", 30),
    ("round(2.5 * 3, 1)", 7.5),
])
def test_evaluate(expression, expected):
    assert numbers_match(evaluate(expression), expected)


@pytest.mark.parametrize("expression", [
    "__import__('os').system('ls')",   # no imports / dunder calls
    "open('secrets.txt')",              # no arbitrary functions
    "x + 1",                            # no variables
    "(1).real",                         # no attribute access
    "2 ** 1000",                        # bounded exponent
    "1 / 0",
    "to_minutes('25:10')",              # invalid time
    "'a' + 'b'",                        # strings only as to_minutes arguments
    "3 +",                              # syntax error
])
def test_evaluate_rejects_unsafe_or_invalid(expression):
    with pytest.raises(CalculationError):
        evaluate(expression)


def test_to_minutes():
    assert to_minutes("00:00") == 0
    assert to_minutes("9:05") == 545
    assert to_minutes("23:59") == 1439


def test_parse_llm_json_accepts_code_fences():
    plan = parse_llm_json('```json\n{"steps": ["a", "b"]}\n```', Plan)
    assert plan.steps == ["a", "b"]
    assert plan.as_text() == "1. a 2. b"


@pytest.mark.parametrize("reply", ["not json", '{"steps": []}', '{"wrong": 1}'])
def test_parse_llm_json_rejects_bad_output(reply):
    with pytest.raises(OutputFormatError):
        parse_llm_json(reply, Plan)


def test_execution_requires_final_answer():
    with pytest.raises(OutputFormatError):
        parse_llm_json('{"steps": [{"step": 1, "description": "x"}], "explanation": "e"}', Execution)
