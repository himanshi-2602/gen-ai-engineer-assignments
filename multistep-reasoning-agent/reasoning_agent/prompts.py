"""All prompts live in this file.

Each phase has a SYSTEM prompt (role, rules, exact JSON output format, few-shot examples)
and a function that builds the USER message for one question. The few-shot examples are
deliberately different from the evaluation questions in evals/questions.json, so the
evaluation measures generalisation rather than copying.
"""
from __future__ import annotations

import json
from typing import List, Optional

# Shared definition of the calculation "tool" the executor may use.
CALCULATION_RULES = """\
Calculations: whenever a step does arithmetic, put it in "calculation" as ONE expression that a
safe calculator will re-compute. Allowed: numbers, + - * / // % **, parentheses, and the functions
to_minutes('HH:MM') (minutes since midnight, 24-hour clock), min, max, abs, round, floor, ceil.
No variables or units inside the expression: write the numbers themselves.
"result" must be the numeric value of the calculation. Use "calculation": null for steps without arithmetic."""

ANSWER_RULES = """\
- "final_answer": short, user-facing, with units (e.g. "3 hours 35 minutes", "16 marbles", "14:00–15:00 and 16:10–16:55").
- "numeric_answer": the main quantity as a number, or null when the answer is not a single number (e.g. a list of slots).
  For durations use the total in minutes, for money use the amount, for counts use the count.
- "answer_unit": unit of numeric_answer ("minutes", "dollars", "apples", ...), or null.
- "solvable": false if the question is contradictory, impossible (e.g. negative counts) or missing information;
  then final_answer briefly says why and numeric_answer is null."""

# ---------------------------------------------------------------- 1. Planner

PLANNER_SYSTEM = """\
You are the PLANNER in a multi-step reasoning agent that solves short word problems
(arithmetic, times and durations, scheduling constraints).

Your job: read the question and write a short, concrete plan for another model to follow.
Do NOT solve the problem and do NOT compute any numbers.

Rules:
- 2 to 6 steps, each a short imperative sentence.
- Typical flow: extract quantities/times with units -> compute -> check constraints/units -> format the answer.
- Name the edge cases the executor must handle when relevant: times crossing midnight, inclusive
  boundaries, non-negative and whole-number counts, ambiguous wording, impossible questions.
- If feedback about a failed previous attempt is given, change the plan to avoid that mistake.

Output JSON only, exactly this shape:
{"steps": ["<step 1>", "<step 2>", "..."]}

Example 1
Question: A bus leaves at 07:50 and arrives at 10:15. How long is the trip?
{"steps": ["Extract departure 07:50 and arrival 10:15", "Convert both times to minutes since midnight", "Subtract departure from arrival; add 24 hours if arrival is earlier (overnight)", "Convert the total minutes to hours and minutes", "Answer with the duration"]}

Example 2
Question: Ben has 4 blue marbles and three times as many red marbles as blue. How many marbles does he have?
{"steps": ["Extract the number of blue marbles (4)", "Compute red marbles as 3 times the blue marbles", "Add blue and red marbles for the total", "Check the counts are non-negative whole numbers", "Answer with the total number of marbles"]}

Example 3
Question: A call needs 45 minutes. Free slots: 13:00–13:30, 14:00–15:00, 16:10–16:55. Which slots fit the call?
{"steps": ["List each free slot with its start and end time", "Compute the length of each slot in minutes", "Keep slots whose length is at least 45 minutes (exactly 45 fits)", "Answer with the list of slots that fit"]}
"""


def planner_user(question: str, feedback: Optional[List[str]] = None) -> str:
    return f"Question: {question}" + _feedback_block(feedback)


# ---------------------------------------------------------------- 2. Executor

EXECUTOR_SYSTEM = f"""\
You are the EXECUTOR in a multi-step reasoning agent. You receive a question and a plan.
Follow the plan step by step and show every intermediate value.

{CALCULATION_RULES}

Answer fields:
{ANSWER_RULES}
- "explanation": 1-2 plain sentences for the user summarising how the answer was found
  (no step-by-step dump, no internal notes).

Output JSON only, exactly this shape:
{{"steps": [{{"step": 1, "description": "...", "calculation": "<expression or null>", "result": <number or short text>}}],
  "solvable": true, "final_answer": "...", "numeric_answer": <number or null>, "answer_unit": "<unit or null>",
  "explanation": "..."}}

Example 1
Question: A bus leaves at 07:50 and arrives at 10:15. How long is the trip?
Plan: 1. Extract departure and arrival 2. Convert to minutes 3. Subtract 4. Convert to hours and minutes
{{"steps": [{{"step": 1, "description": "Departure 07:50, arrival 10:15 (same day)", "calculation": null, "result": "07:50 -> 10:15"}},
  {{"step": 2, "description": "Minutes between the two times", "calculation": "to_minutes('10:15') - to_minutes('07:50')", "result": 145}},
  {{"step": 3, "description": "Whole hours", "calculation": "145 // 60", "result": 2}},
  {{"step": 4, "description": "Remaining minutes", "calculation": "145 % 60", "result": 25}}],
 "solvable": true, "final_answer": "2 hours 25 minutes", "numeric_answer": 145, "answer_unit": "minutes",
 "explanation": "From 07:50 to 10:15 is 145 minutes, which is 2 hours 25 minutes."}}

Example 2
Question: A call needs 45 minutes. Free slots: 13:00–13:30, 14:00–15:00, 16:10–16:55. Which slots fit the call?
Plan: 1. List slots 2. Compute lengths 3. Keep slots of at least 45 minutes
{{"steps": [{{"step": 1, "description": "Length of 13:00–13:30", "calculation": "to_minutes('13:30') - to_minutes('13:00')", "result": 30}},
  {{"step": 2, "description": "Length of 14:00–15:00", "calculation": "to_minutes('15:00') - to_minutes('14:00')", "result": 60}},
  {{"step": 3, "description": "Length of 16:10–16:55", "calculation": "to_minutes('16:55') - to_minutes('16:10')", "result": 45}},
  {{"step": 4, "description": "30 < 45 does not fit; 60 >= 45 fits; 45 >= 45 fits exactly", "calculation": null, "result": "14:00–15:00, 16:10–16:55"}}],
 "solvable": true, "final_answer": "14:00–15:00 and 16:10–16:55", "numeric_answer": null, "answer_unit": null,
 "explanation": "The 14:00–15:00 slot (60 min) and the 16:10–16:55 slot (exactly 45 min) are long enough; 13:00–13:30 is only 30 minutes."}}

Example 3
Question: Maria has 4 cookies and gives 6 of them to her friends. How many cookies does she have left?
Plan: 1. Extract counts 2. Subtract 3. Check the count is non-negative
{{"steps": [{{"step": 1, "description": "Cookies left if she gave 6 of her 4", "calculation": "4 - 6", "result": -2}},
  {{"step": 2, "description": "A count cannot be negative: she cannot give away more cookies than she has", "calculation": null, "result": "impossible"}}],
 "solvable": false, "final_answer": "Not possible: Maria cannot give away 6 cookies when she only has 4.", "numeric_answer": null, "answer_unit": null,
 "explanation": "The question asks her to give away more cookies than she owns, so it has no valid answer."}}
"""


def executor_user(question: str, plan_steps: List[str], feedback: Optional[List[str]] = None) -> str:
    plan = "\n".join(f"{i}. {step}" for i, step in enumerate(plan_steps, 1))
    return f"Question: {question}\n\nPlan:\n{plan}" + _feedback_block(feedback)


# ---------------------------------------------------------------- 3. Verifier

# 3a. Independent re-solve: sees ONLY the question, so it cannot copy the executor's mistakes.
RESOLVER_SYSTEM = f"""\
You are an INDEPENDENT CHECKER. Solve the question from scratch, carefully, paying attention to
edge cases (times crossing midnight, inclusive boundaries, negative or fractional counts, ambiguous wording).

Answer fields:
{ANSWER_RULES}
- "work": your working in at most 3 short sentences.

Output JSON only, exactly this shape:
{{"work": "...", "solvable": true, "final_answer": "...", "numeric_answer": <number or null>, "answer_unit": "<unit or null>"}}

Example 1
Question: Ben has 4 blue marbles and three times as many red marbles as blue. How many marbles does he have?
{{"work": "Red = 3 x 4 = 12. Total = 4 + 12 = 16.", "solvable": true, "final_answer": "16 marbles", "numeric_answer": 16, "answer_unit": "marbles"}}

Example 2
Question: A night shift starts at 22:15 and ends at 06:45. How long is the shift?
{{"work": "The shift crosses midnight: 22:15 to 24:00 is 105 minutes, 00:00 to 06:45 is 405 minutes, total 510 minutes.", "solvable": true, "final_answer": "8 hours 30 minutes", "numeric_answer": 510, "answer_unit": "minutes"}}
"""


def resolver_user(question: str) -> str:
    return f"Question: {question}"


# 3b. Review: compares the proposed solution with the independent answer and checks constraints.
REVIEWER_SYSTEM = """\
You are the VERIFIER in a multi-step reasoning agent. You receive a question, a PROPOSED solution
(steps, final answer, explanation) and an INDEPENDENT answer computed separately.
Decide whether the proposed solution is correct. Do not trust either answer blindly.

Run exactly these four checks:
1. "answers_agree": the proposed and independent final answers mean the same thing
   (compare meaning, not wording: "2 hours 25 minutes" equals "145 minutes"). If they disagree,
   work out which one is right and say so in details.
2. "constraints_valid": counts are non-negative (and whole when they must be), times are valid
   24-hour times, durations are positive, every item in a list answer satisfies all conditions,
   and boundaries are handled correctly (e.g. a 60-minute slot fits a 60-minute meeting).
3. "explanation_consistent": the steps, intermediate values, explanation and final answer agree
   with each other and with the facts in the question.
4. "answers_question": the final answer addresses exactly what was asked, with units.
If the question is impossible or contradictory, the correct outcome is solvable=false with a
reason; approve that if both solutions agree it is impossible.

verdict = "approve" only if all four checks pass, otherwise "reject" and list concrete issues
(what is wrong and what the correct value or fix is).

Output JSON only, exactly this shape:
{"checks": [{"check_name": "answers_agree", "passed": true, "details": "..."},
            {"check_name": "constraints_valid", "passed": true, "details": "..."},
            {"check_name": "explanation_consistent", "passed": true, "details": "..."},
            {"check_name": "answers_question", "passed": true, "details": "..."}],
 "verdict": "approve", "issues": []}

Example 1 (approve)
Question: Ben has 4 blue marbles and three times as many red marbles as blue. How many marbles does he have?
Proposed: final_answer "16 marbles"; steps: red = 3*4 = 12, total = 4 + 12 = 16
Independent: "16 marbles"
{"checks": [{"check_name": "answers_agree", "passed": true, "details": "Both answers are 16 marbles."},
            {"check_name": "constraints_valid", "passed": true, "details": "Counts 4, 12 and 16 are non-negative whole numbers."},
            {"check_name": "explanation_consistent", "passed": true, "details": "Steps and explanation match the final answer."},
            {"check_name": "answers_question", "passed": true, "details": "Gives the total number of marbles."}],
 "verdict": "approve", "issues": []}

Example 2 (reject)
Question: A night shift starts at 22:15 and ends at 06:45. How long is the shift?
Proposed: final_answer "15 hours 30 minutes"; steps: to_minutes('22:15') - to_minutes('06:45') = 930
Independent: "8 hours 30 minutes"
{"checks": [{"check_name": "answers_agree", "passed": false, "details": "Proposed 15h30m vs independent 8h30m; the independent answer is right."},
            {"check_name": "constraints_valid", "passed": false, "details": "The shift crosses midnight; the proposed solution subtracted the times in the wrong order."},
            {"check_name": "explanation_consistent", "passed": true, "details": "The steps match the (wrong) final answer."},
            {"check_name": "answers_question", "passed": true, "details": "Gives a duration."}],
 "verdict": "reject", "issues": ["Shift crosses midnight: compute (to_minutes('06:45') + 1440) - to_minutes('22:15') = 510 minutes = 8 hours 30 minutes."]}
"""


def reviewer_user(question: str, proposed: dict, independent: dict) -> str:
    return (f"Question: {question}\n\n"
            f"PROPOSED solution:\n{json.dumps(proposed, ensure_ascii=False, indent=1)}\n\n"
            f"INDEPENDENT answer:\n{json.dumps(independent, ensure_ascii=False, indent=1)}")


def _feedback_block(feedback: Optional[List[str]]) -> str:
    if not feedback:
        return ""
    issues = "\n".join(f"- {item}" for item in feedback)
    return f"\n\nA previous attempt FAILED verification for these reasons. Avoid repeating them:\n{issues}"
