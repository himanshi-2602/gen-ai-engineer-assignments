"""Run the evaluation questions through the real agent and log the results.

    .venv/bin/python evals/run_evals.py            # all questions
    .venv/bin/python evals/run_evals.py E1 T6      # selected questions
    .venv/bin/python evals/run_evals.py --missing  # only questions without a valid result yet

Results are merged into the existing log by question id, so a run interrupted by API quota
limits can be completed later without repeating finished questions.

For each question it logs: the question, the final JSON, whether the verifier passed, whether
the agent retried, and whether the answer matches the expected one (simple regex grading).
Writes evals/results/run_log.jsonl and evals/results/summary.md.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reasoning_agent.agent import build_agent  # noqa: E402
from reasoning_agent.config import Settings  # noqa: E402


def grade(case: dict, response: dict) -> bool:
    """Heuristic check of the answer text against the expected patterns."""
    text = f"{response['answer']} {response['reasoning_visible_to_user']}".lower()
    answer = response["answer"].lower()
    if response["status"] != case.get("expected_status", "success"):
        return False
    if case.get("must_match_any") and not any(re.search(p, text) for p in case["must_match_any"]):
        return False
    if any(not re.search(p, answer) for p in case.get("must_match_all", [])):
        return False
    return not any(re.search(p, answer) for p in case.get("must_not_match", []))


def main() -> None:
    all_cases = json.loads((ROOT / "evals/questions.json").read_text())
    out_dir = ROOT / "evals/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run_log.jsonl"
    previous = {}
    if log_path.exists():
        previous = {r["id"]: r for r in map(json.loads, log_path.read_text().splitlines()) if r}

    args = sys.argv[1:]
    if args == ["--missing"]:
        cases = [c for c in all_cases if c["id"] not in previous or previous[c["id"]].get("llm_error")]
    elif args:
        cases = [c for c in all_cases if c["id"] in set(args)]
    else:
        cases = all_cases

    settings = Settings.from_env()
    agent = build_agent(settings)
    rows = []

    for case in cases:
        start = time.monotonic()
        response, trace = agent.solve_with_trace(case["question"])
        data = response.model_dump()
        row = {
            "id": case["id"], "category": case["category"], "question": case["question"],
            "expected": case["expected"], "final_json": data,
            "verifier_passed": trace["verifier_passed"], "retried": data["metadata"]["retries"] > 0,
            "retries": data["metadata"]["retries"], "correct": grade(case, data),
            "seconds": round(time.monotonic() - start, 1),
            "model": getattr(agent.planner.llm, "model", None),
            # The LLM could not be reached (e.g. quota): not a reasoning result, rerun later.
            "llm_error": any("error" in a and "invalid" not in a["error"] for a in trace["attempts"]),
        }
        rows.append(row)
        print(f"{row['id']:3} {'OK  ' if row['correct'] else 'MISS'} verifier={'pass' if row['verifier_passed'] else 'FAIL'} "
              f"retries={row['retries']} [{data['status']}] {data['answer']}", flush=True)

    merged = {**previous, **{r["id"]: r for r in rows}}
    rows = [merged[c["id"]] for c in all_cases if c["id"] in merged]
    with log_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_summary(rows, settings, out_dir / "summary.md")
    correct = sum(r["correct"] for r in rows)
    errors = sum(r.get("llm_error", False) for r in rows)
    print(f"\n{correct}/{len(rows)} correct, {errors} with LLM errors. "
          f"Logs: evals/results/run_log.jsonl, evals/results/summary.md")


def write_summary(rows, settings, path: Path) -> None:
    lines = [
        "# Evaluation run",
        "",
        f"Models (fallback order): `{settings.gemini_model}` | max retries: {settings.max_retries} | "
        f"questions: {len(rows)} | correct: {sum(r['correct'] for r in rows)}/{len(rows)} | "
        f"verifier passed: {sum(r['verifier_passed'] for r in rows)}/{len(rows)} | "
        f"retried: {sum(r['retried'] for r in rows)}/{len(rows)}",
        "",
        "| ID | Type | Question | Answer | Status | Verifier passed | Retries | Correct | Model | Time |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        d = r["final_json"]
        lines.append(f"| {r['id']} | {r['category']} | {r['question']} | {d['answer']} | {d['status']} "
                     f"| {'yes' if r['verifier_passed'] else 'no'} | {r['retries']} "
                     f"| {'✅' if r['correct'] else '❌'} | {r.get('model') or '?'} | {r['seconds']}s |")
    lines += ["", "## Full JSON per question", ""]
    for r in rows:
        lines += [f"### {r['id']}: {r['question']}", "",
                  f"Expected: {r['expected']} · verifier passed: {r['verifier_passed']} · retried: {r['retried']}",
                  "", "```json", json.dumps(r["final_json"], indent=2, ensure_ascii=False), "```", ""]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
