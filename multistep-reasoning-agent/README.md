# Multi-Step Reasoning Agent with Self-Checking

A small reasoning agent that solves word problems (arithmetic, times and durations, scheduling constraints) in multiple steps: a **planner**, an **executor** (LLM plus a Python calculator) and a **verifier** (code checks, an independent re-solve and an LLM review). It **retries** with feedback when verification fails and shows the user only the final answer and a short explanation.

**Result:** **15/15** evaluation questions correct (8 easy, 7 tricky), verified by the agent's own checks. One question needed a retry, which the verifier triggered correctly. See [Evaluation](#evaluation-and-example-run-logs).

---

## Problem

Build a reasoning agent that solves structured word problems in multiple steps, checks its own work, and shows only the final answer to the user. Any LLM API and any language may be used.

- **Input:** a plain-text question, e.g. *"If a train leaves at 14:30 and arrives at 18:05, how long is the journey?"*
- **Output:** a JSON object:
  ```json
  {
    "answer": "<final short answer, user-facing>",
    "status": "success" | "failed",
    "reasoning_visible_to_user": "<short explanation, but no raw chain-of-thought logs>",
    "metadata": {
      "plan": "<model's internal plan, can be abbreviated>",
      "checks": [{"check_name": "<string>", "passed": true, "details": "<string>"}],
      "retries": <integer>
    }
  }
  ```
- **Agent:** at least three phases. The **Planner** writes a step-by-step plan. The **Executor** follows it, producing intermediate values; it may use the LLM and/or code. The **Verifier** re-solves independently, validates constraints and/or looks for inconsistencies. On failure, the agent retries a limited number of times or returns `"failed"` with the reason. The full chain-of-thought must not be shown to the user.
- **Interface:** a CLI, an HTTP endpoint (`/solve`) or a notebook function `solve(question) -> dict`.
- **Technical:** real or mocked LLM calls, with a swappable design; prompts not hard-coded everywhere; phases separated.
- **Prompts:** separate planner, executor and verifier prompts, each with an output format and 2–3 examples.
- **Evaluation:** 5–10 easy and 3–5 tricky questions, logging the question, the final JSON, whether the verifier passed and whether the agent retried.
- **Submit:** the code with a README (how to run, where the prompts live, assumptions), prompt documentation (design, what didn't work, improvements) and example run logs.

---

## Requirements checklist

| PDF requirement | Where it is |
|---|---|
| Plain-text question → JSON with the exact schema | [`reasoning_agent/schemas.py`](reasoning_agent/schemas.py) (`SolveResponse`); a test checks the exact keys |
| **Planner** (step-by-step plan) | `Planner` in [`reasoning_agent/phases.py`](reasoning_agent/phases.py) |
| **Executor** (intermediate values; LLM + code) | `Executor` in `phases.py`, plus the safe calculator in [`reasoning_agent/tools.py`](reasoning_agent/tools.py) |
| **Verifier**: re-solve independently ✅, validate constraints ✅, look for inconsistencies ✅ (all three, not just one) | `Verifier` in `phases.py` |
| Retry a limited number of times, otherwise `"failed"` with the reason | Loop in [`reasoning_agent/agent.py`](reasoning_agent/agent.py) (`MAX_RETRIES=2`, so 3 attempts) |
| Chain-of-thought not shown to the user; kept in metadata or logs | The response has only the answer and a 1–2 sentence explanation. Full steps go to `logs/trace.jsonl` |
| Interface | **All three:** CLI `agent.py`, HTTP `POST /solve` in `api.py` (plus a small web page), and `from reasoning_agent import solve` |
| Real or mocked LLM, swappable | Real **Gemini** and a **MockLLM** behind one `LLMClient` interface ([`reasoning_agent/llm.py`](reasoning_agent/llm.py)) |
| Prompts not hard-coded everywhere | All prompts are in **one file**: [`reasoning_agent/prompts.py`](reasoning_agent/prompts.py) |
| Planner, executor and verifier separated | One class each in `phases.py`; orchestration in `agent.py` |
| Separate planner / executor / verifier prompts, with output format and 2–3 examples | `prompts.py`: 4 prompts (the verifier has 2), each with a JSON format and 2–3 examples; a test checks this |
| 5–10 easy and 3–5+ tricky test questions | [`evals/questions.json`](evals/questions.json): **8 easy + 7 tricky** |
| Each test logs question, final JSON, verifier passed, retried | [`evals/results/run_log.jsonl`](evals/results/run_log.jsonl) and [`summary.md`](evals/results/summary.md) |
| README: how to run, where prompts live, assumptions | [How to run](#how-to-run), [Where the prompts live](#where-the-prompts-live), [Assumptions](#assumptions) |
| Prompt documentation: why this design, what didn't work, improvements | [Prompt documentation](#prompt-documentation) |
| Example run logs (5–10 questions) | [Evaluation](#evaluation-and-example-run-logs): 15 questions with full JSON, plus one full internal trace |

---

## How it works

```
question ──► PLANNER (LLM) ──► numbered plan (no solving)
                 │
                 ▼
             EXECUTOR (LLM) ──► steps with intermediate values; every calculation written
                 │              as an expression, e.g. to_minutes('18:05') - to_minutes('14:30')
                 ▼
             VERIFIER
               1. arithmetic_recomputed      (code)  re-computes every expression with a safe Python calculator
               2. independent_resolve_match  (LLM + code)  a fresh call solves the question WITHOUT seeing
                                                     the proposed solution; numeric answers compared in code
               3. reviewer (LLM): answers_agree, constraints_valid, explanation_consistent,
                                  answers_question, then a verdict
                 │
        all checks pass? ── yes ──► response (success; or "failed" if the question is verifiably impossible)
                 │ no
                 ▼
        retry (max 2) with the verifier's issues fed back to the planner and executor
                 │ still failing
                 ▼
        response: status "failed", with the reason and the last unverified answer
```

Each attempt makes **4 LLM calls**: planner, executor, independent re-solve and reviewer.

**What the user sees** (from the CLI with `--user`, or from the web page):
```
Answer: 11:00–12:00  [success]
Why: Only the 11:00–12:00 slot provides the required 60 minutes for the meeting; the other slots are too short.
```
The plan, the checks and the retry count are in `metadata`, for debugging and evaluation. The executor's step-by-step working, the independent solution and the reviewer's output are written only to `logs/trace.jsonl`. See [`evals/results/example_trace_T1_retry.json`](evals/results/example_trace_T1_retry.json) for a complete trace.

---

## How to run

### 1. Setup (Python 3.9+)
```bash
cd multistep-reasoning-agent
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 2. Run the tests (no API key needed)
```bash
.venv/bin/python -m pytest -q          # 72 tests, all offline, using MockLLM
```

### 3. Add an LLM key (needed to solve real questions)
The agent uses **Google Gemini**. Get a free key at <https://aistudio.google.com/apikey> (use "Create API key in new project", without billing, for the free tier). Then:
```bash
cp .env.example .env
# edit .env and set GEMINI_API_KEY=<your key>
.venv/bin/python scripts/hello_llm.py      # checks the key works
```
**No API key is included in this repository.** `.env` is listed in `.gitignore`; only `.env.example`, with a placeholder, is committed.

### 4. Use the agent
```bash
# CLI: interactive (type questions; an empty line quits) or a single question
.venv/bin/python agent.py
.venv/bin/python agent.py "A train leaves at 22:40 and arrives at 01:15 the next day. How long is the journey?"
.venv/bin/python agent.py --user "What is 15% of 240?"          # only the user-facing fields

# HTTP API
.venv/bin/uvicorn api:app --port 8000
#   http://localhost:8000/        simple web page: type a question, click Solve
#   http://localhost:8000/docs    interactive API docs (POST /solve → Try it out)
curl -X POST localhost:8000/solve -H "Content-Type: application/json" \
     -d '{"question": "Alice has 3 red apples and twice as many green apples as red. How many apples does she have in total?"}'

# Python / notebook
.venv/bin/python -c "from reasoning_agent import solve; print(solve('What is 15% of 240?'))"
```
Each question takes about 20–60 seconds: 4 LLM calls, spaced to respect free-tier rate limits (`LLM_MIN_INTERVAL`).

### 5. Run the evaluation
```bash
.venv/bin/python evals/run_evals.py             # all 15 questions → evals/results/
.venv/bin/python evals/run_evals.py T1 T6       # selected questions
.venv/bin/python evals/run_evals.py --missing   # only questions without a valid result (resumes after quota errors)
```

### Configuration (`.env`)
| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | (none) | Your key (required for real runs) |
| `GEMINI_MODEL` | `gemini-3.5-flash,gemini-3-flash-preview,gemini-3.1-flash-lite,gemini-3.5-flash-lite` | **Fallback list**: if a model's daily quota is used up or it stays overloaded, the next one is used |
| `MAX_RETRIES` | `2` | Extra plan, execute and verify attempts after a failed verification |
| `LLM_MIN_INTERVAL` | `6` | Seconds between LLM calls (free-tier per-minute limits) |
| `TRACE_PATH` | `logs/trace.jsonl` | Where the full internal trace of every run is written |

### How a reviewer can check this without my key
1. **Unit tests** (offline, about 0.5 s): `pytest` exercises every branch of the agent loop with a scripted mock LLM: success, reviewer rejection then retry, invalid JSON then retry, arithmetic caught by code, retries exhausted then `failed`, an impossible question, an LLM outage, the API, and model fallback.
2. **Recorded real runs:** [`evals/results/summary.md`](evals/results/summary.md) has the full JSON for all 15 questions, and [`example_trace_T1_retry.json`](evals/results/example_trace_T1_retry.json) shows the hidden internal trace of a run that retried.
3. **Live:** with your own free Gemini key in `.env`, run steps 3–5 above.

---

## Where the prompts live

All prompts are in **[`reasoning_agent/prompts.py`](reasoning_agent/prompts.py)**. Nothing else contains prompt text.

| Prompt | Used by | Output format |
|---|---|---|
| `PLANNER_SYSTEM` + `planner_user()` | Planner | `{"steps": [...]}` |
| `EXECUTOR_SYSTEM` + `executor_user()` | Executor | `{"steps": [{"step", "description", "calculation", "result"}], "solvable", "final_answer", "numeric_answer", "answer_unit", "explanation"}` |
| `RESOLVER_SYSTEM` + `resolver_user()` | Verifier, independent re-solve | `{"work", "solvable", "final_answer", "numeric_answer", "answer_unit"}` |
| `REVIEWER_SYSTEM` + `reviewer_user()` | Verifier, review | `{"checks": [4 named checks], "verdict": "approve"\|"reject", "issues": [...]}` |

Shared rules (`CALCULATION_RULES`, `ANSWER_RULES`) are defined once and included in several prompts. Each system prompt has **2–3 worked examples**. A unit test checks that every prompt specifies "Output JSON only" and has at least 2 examples.

---

## Prompt documentation

### Why the prompts are designed this way
- **One role per prompt.** The planner is told *not* to solve or compute, so planning stays separate from execution and can be inspected. The executor only follows the plan. The verifier has two separate prompts with different jobs.
- **Strict JSON, validated with Pydantic.** Every prompt states the exact JSON shape, and the API is called in JSON mode (`response_mime_type="application/json"`). Every reply is validated against a Pydantic model; a malformed reply counts as a failed attempt (`output_format` check) and is retried, instead of crashing or being guessed at.
- **Calculations as expressions, checked in code.** LLMs can slip on arithmetic, so the executor must write each calculation as an expression (`to_minutes('18:05') - to_minutes('14:30')`) next to its result. The verifier re-computes it with a **safe AST-based calculator**: only arithmetic and a whitelist of functions are allowed, with no variables, attributes or imports, so model output can never run arbitrary code. `to_minutes` validates times (00:00–23:59).
- **Independent re-solve without the proposed answer.** If the checker sees the proposed answer, it tends to agree with it (anchoring). So the re-solve prompt receives **only the question**, and a separate reviewer prompt compares the two answers. Numeric answers are also compared in code (`numeric_answer` plus `answer_unit`), so agreement doesn't depend only on an LLM's opinion.
- **Explicit constraint checklist for the reviewer.** It names four checks (`answers_agree`, `constraints_valid`, `explanation_consistent`, `answers_question`) and lists the edge cases to look for: midnight crossing, inclusive boundaries, negative or fractional counts, invalid times. This gives a consistent `checks` list in the metadata.
- **"Solvable" as a first-class outcome.** Impossible questions (e.g. giving away more apples than you have) shouldn't produce "-2 apples". The executor and re-solver can set `solvable: false`; when both agree, the agent returns `status: "failed"` with the reason (evaluation question T6).
- **Few-shot examples kept separate from the evaluation questions.** The prompt examples (a bus trip, marbles, a 45-minute call, a night shift, cookies) are deliberately *different* from the evaluation questions, so the results measure generalisation rather than copying.
- **A short user-facing explanation.** The executor writes a separate 1–2 sentence `explanation` for the user, so the step-by-step working never needs to be shown.
- **Feedback on retry.** When verification fails, the verifier's concrete issues ("step 2: invalid time '24:00'") are added to the next planner and executor prompts, so a retry isn't just the same attempt again.
- **Temperature 0** everywhere, for repeatable answers.

### What I tried that didn't work well
These happened during development and are recorded in the logs:
1. **Default model retired.** I started with `gemini-2.5-flash`; the API returned *404: no longer available to new users*. I switched to the current Gemini Flash models.
2. **The newest model was overloaded.** `gemini-3.8-flash` (the model the API recommended) returned **503 "high demand"** and **504 timeouts** in testing. One request with no timeout set **hung for over 5 minutes**. Fixes: a 60-second timeout on every call, retries with exponential backoff for 429/5xx errors, and switching to the next model after repeated failures.
3. **Free-tier quota.** The free tier allows **20 requests per day per model**, and each question needs 4 calls (about 5 questions per model per day). The first evaluation run stopped after 2 questions with `429 RESOURCE_EXHAUSTED`. Fixes: a **model fallback list** (each model has its own daily quota), pacing between calls, and a `--missing` mode so an interrupted evaluation resumes without repeating finished questions. In the recorded run, E1–E2 were answered by `gemini-3.5-flash` and E3–T7 by the fallback `gemini-3.1-flash-lite`.
4. **The executor used `to_minutes('24:00')`.** On T1 (overnight train), the first attempt computed the time to midnight as `to_minutes('24:00') - to_minutes('22:40')`. 24:00 isn't a valid 24-hour time, so the code check failed it. The issue was fed back, and the retry wrote midnight as `1440 - to_minutes('22:40')`, giving the correct 2 hours 35 minutes. The verifier and retry worked as designed, but the prompt could prevent this in the first place (see the improvements below).
5. **A key from a billing-enabled project** returned *402: prepayment credits depleted* for every model: the free tier doesn't apply once billing is enabled. The client now explains this error in plain words.

### What I would improve with more time
- **Prompt fix for midnight:** add to `CALCULATION_RULES`: "midnight is 1440 minutes; never write `to_minutes('24:00')`". This would avoid T1's retry. I didn't change the prompt after the evaluation, so the recorded logs match the code.
- **A larger, harder test set** (50+ questions, including ambiguous ones where the right behaviour is to ask or to state an assumption), with repeated runs to measure consistency. The current set is small and every question passed, so it can't show weak spots.
- **Fewer LLM calls:** skip the reviewer call when the code checks already prove agreement (identical numeric answers, all arithmetic verified). That's 3 calls instead of 4, which matters with free-tier quotas.
- **Different model for the independent re-solve**, so the checker doesn't share the executor's blind spots.
- **Native tool or function calling**, letting the model call the calculator during execution rather than only having its results verified afterwards.
- **Semantic answer comparison in code** for lists and times (e.g. parse slot lists), so less is left to the LLM reviewer.
- **Async LLM calls and caching**, and per-request cost and latency metrics.

---

## Assumptions
- Times use the **24-hour clock** (HH:MM). An arrival time earlier than the departure time means the next day.
- A slot fits a meeting if its length is **at least** the meeting length (a boundary match fits).
- Counts of objects must be non-negative. A question that would require a negative count is reported as **impossible** (`status: "failed"` with the reason) rather than answered.
- "From 9 to 5" for opening hours means 9:00 to 17:00.
- Answers are reported with units. For durations, the internal `numeric_answer` is in minutes.
- `status: "failed"` covers both "couldn't verify an answer after the retries" and "verified that the question has no valid answer". `reasoning_visible_to_user` says which.
- `plan` in the metadata is the final attempt's plan. `checks` includes the checks from **every** attempt (labelled `[attempt N]` when there was a retry), so retries are visible.

---

## Evaluation and example run logs

Run on 2026-11-03 with the model fallback list above (max retries 2). Full JSON for every question: [`evals/results/summary.md`](evals/results/summary.md). Machine-readable log: [`run_log.jsonl`](evals/results/run_log.jsonl).

**15/15 correct · verifier passed 15/15 · retried 1/15**

| ID | Type | Question | Answer | Status | Verifier passed | Retries | Correct |
|---|---|---|---|---|---|---|---|
| E1 | easy | If a train leaves at 14:30 and arrives at 18:05, how long is the journey? | 3 hours 35 minutes | success | yes | 0 | ✅ |
| E2 | easy | Alice has 3 red apples and twice as many green apples as red. How many apples does she have in total? | 9 apples | success | yes | 0 | ✅ |
| E3 | easy | A meeting needs 60 minutes. There are free slots: 09:00–09:30, 09:45–10:30, 11:00–12:00. Which slots can fit the meeting? | 11:00–12:00 | success | yes | 0 | ✅ |
| E4 | easy | What is 15% of 240? | 36 | success | yes | 0 | ✅ |
| E5 | easy | A shop sells pens at $3 each. How much do 7 pens cost? | 21 dollars | success | yes | 0 | ✅ |
| E6 | easy | Tom is 12 years old and his sister is 5 years older. How old will his sister be in 3 years? | 20 years old | success | yes | 0 | ✅ |
| E7 | easy | A movie starts at 19:45 and lasts 2 hours 20 minutes. What time does it end? | 22:05 | success | yes | 0 | ✅ |
| E8 | easy | A runner covers 12 km in 1 hour 30 minutes. What is the average speed in km per hour? | 8 km/h | success | yes | 0 | ✅ |
| T1 | tricky (crosses midnight) | A train leaves at 22:40 and arrives at 01:15 the next day. How long is the journey? | 2 hours 35 minutes | success | yes | **1** | ✅ |
| T2 | tricky (intuitive trap) | A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? | $0.05 | success | yes | 0 | ✅ |
| T3 | tricky (boundary) | A meeting needs 30 minutes. Free slots: 09:00–09:30, 10:00–10:20, 13:15–13:45. Which slots can fit the meeting? | 09:00–09:30 and 13:15–13:45 | success | yes | 0 | ✅ |
| T4 | tricky (multi-step) | Sam has 5 apples, eats 2, buys a dozen more, then gives a third of his apples to a friend. How many apples does Sam have now? | 10 apples | success | yes | 0 | ✅ |
| T5 | tricky (mixed number) | A recipe needs 2 1/2 cups of flour per batch. How many cups of flour are needed for 3 batches? | 7.5 cups | success | yes | 0 | ✅ |
| T6 | tricky (impossible) | Alice has 3 apples and gives away 5 of them. How many apples does she have left? | Not possible: Alice cannot give away 5 apples when she only has 3. | failed (expected) | yes | 0 | ✅ |
| T7 | tricky (ambiguous "9 to 5") | A shop is open from 9 to 5 every day. How many hours is it open each day? | 8 hours | success | yes | 0 | ✅ |

"Correct" is graded automatically with simple patterns ([`run_evals.py`](evals/run_evals.py) `grade()`); I also checked every answer by hand. A test checks that the grader accepts the expected answers and rejects common wrong ones (e.g. "$0.10" for T2).

**Example output** (E3, full JSON as returned by `/solve`):
```json
{
  "answer": "11:00–12:00",
  "status": "success",
  "reasoning_visible_to_user": "Only the 11:00–12:00 slot provides the required 60 minutes for the meeting; the other slots are too short.",
  "metadata": {
    "plan": "1. Identify the duration of the meeting (60 minutes) 2. Calculate the duration of each free slot by subtracting the start time from the end time 3. Compare each slot's duration to the required 60 minutes 4. Select only the slots where the duration is greater than or equal to 60 minutes 5. List the qualifying slots as the final answer",
    "checks": [
      {"check_name": "arithmetic_recomputed", "passed": true, "details": "re-computed 3 calculation(s) in Python; all match"},
      {"check_name": "independent_resolve_match", "passed": true, "details": "answer is not a single number; compared by the reviewer (independent answer: '11:00–12:00')"},
      {"check_name": "answers_agree", "passed": true, "details": "Both the proposed and independent answers identify 11:00–12:00 as the only suitable slot."},
      {"check_name": "constraints_valid", "passed": true, "details": "The durations are calculated correctly and compared against the 60-minute requirement."},
      {"check_name": "explanation_consistent", "passed": true, "details": "The steps and logic clearly show that only the 60-minute slot meets the requirement."},
      {"check_name": "answers_question", "passed": true, "details": "The answer correctly identifies the slot that fits the meeting."},
      {"check_name": "reviewer_verdict", "passed": true, "details": "approved"}
    ],
    "retries": 0
  }
}
```

**Example retry (T1):** the first attempt's executor wrote `to_minutes('24:00') - to_minutes('22:40')`. The code check failed it (*"step 2: invalid time '24:00'; expected HH:MM between 00:00 and 23:59"*), and that issue was fed back into attempt 2, which passed all checks. The response's `checks` show both attempts, and `retries` is 1. The full hidden trace (both plans, all intermediate steps, the independent solutions and the reviewer output) is in [`evals/results/example_trace_T1_retry.json`](evals/results/example_trace_T1_retry.json).

---

## Security
- **API keys are never committed.** They're read from `.env` (git-ignored) or from environment variables; `.env.example` has only a placeholder.
- LLM output is never executed as code. Calculations go through a whitelist-based AST evaluator, with tests that try `__import__`, `open()`, attribute access and huge exponents.
- API input is validated (non-empty, at most 1,000 characters). Invalid requests get `422`; a missing key gives `503`, with a clear message at `/health` and `/solve` instead of a crash.

## Deployment (optional)
- **Render** (free plan): [`render.yaml`](render.yaml) defines the web service (`uvicorn api:app`). Connect the GitHub repo in Render and enter `GEMINI_API_KEY` as a **secret** in the dashboard; it's never stored in the repo.
- **Docker:** a [`Dockerfile`](Dockerfile) is included (`docker build -t reasoning-agent . && docker run -p 8000:8000 -e GEMINI_API_KEY=... reasoning-agent`). *It wasn't tested, because Docker wasn't available on the development machine.*

## Project structure
```
multistep-reasoning-agent/
├── reasoning_agent/
│   ├── config.py        # Settings from environment / .env
│   ├── llm.py           # LLMClient interface, GeminiClient (retries, fallback), MockLLM
│   ├── prompts.py       # ALL prompts: planner, executor, verifier (re-solve + review)
│   ├── schemas.py       # Pydantic models for LLM outputs and the API; JSON parsing
│   ├── tools.py         # Safe calculator (AST whitelist), to_minutes()
│   ├── phases.py        # Planner, Executor, Verifier (+ code checks)
│   └── agent.py         # Agent loop with retries, trace logging, build_agent(), solve()
├── agent.py             # CLI
├── api.py               # FastAPI: POST /solve, GET /health, GET / (web page), /docs
├── evals/
│   ├── questions.json   # 8 easy + 7 tricky questions with expected answers
│   ├── run_evals.py     # Runs them and logs question, JSON, verifier passed, retried
│   └── results/         # run_log.jsonl, summary.md, example_trace_T1_retry.json
├── tests/               # 72 offline unit tests (MockLLM)
├── scripts/hello_llm.py # Check that the API key works
├── .env.example         # Configuration template (no secrets)
├── requirements.txt, Dockerfile, render.yaml
```
