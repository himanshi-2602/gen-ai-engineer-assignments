"""Milestone 0: check that your API key works.

Run from the project folder:
    .venv/bin/python scripts/hello_llm.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reasoning_agent.config import Settings  # noqa: E402
from reasoning_agent.llm import make_llm  # noqa: E402

settings = Settings.from_env()
llm = make_llm(settings)
print(f"Provider: {settings.llm_provider} | model: {settings.gemini_model}")
print("Reply:", llm.generate(
    system="You are a concise assistant.",
    user="In one sentence: what is 17 * 23? Show the result only.",
))
print("JSON reply:", llm.generate(
    system="Reply with JSON only.",
    user='Return {"capital": <capital of France>, "population_millions": <approximate number>}',
    json_mode=True,
))
