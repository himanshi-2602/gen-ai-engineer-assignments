"""Command-line interface.

    python agent.py                       # interactive: type questions, empty line to quit
    python agent.py "What is 15% of 240?"  # one question
    python agent.py --user "..."           # show only what the user would see
"""
from __future__ import annotations

import argparse
import json

from reasoning_agent.agent import build_agent
from reasoning_agent.llm import LLMError


def show(agent, question: str, user_view: bool) -> None:
    response = agent.solve(question).model_dump()
    if user_view:
        print(f"\nAnswer: {response['answer']}  [{response['status']}]")
        print(f"Why: {response['reasoning_visible_to_user']}\n")
    else:
        print(json.dumps(response, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-step reasoning agent with self-checking.")
    parser.add_argument("question", nargs="?", help="question to solve (omit for interactive mode)")
    parser.add_argument("--user", action="store_true", help="print only the user-facing fields")
    args = parser.parse_args()

    try:
        agent = build_agent()
    except LLMError as err:
        raise SystemExit(f"Error: {err}")

    if args.question:
        show(agent, args.question, args.user)
        return
    print("Multi-step reasoning agent. Type a question (empty line to quit).")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        show(agent, question, args.user)


if __name__ == "__main__":
    main()
