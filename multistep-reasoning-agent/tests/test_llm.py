"""Milestone 1 tests. Run: .venv/bin/python -m pytest tests/test_llm.py -v"""
import pytest

from reasoning_agent.config import Settings
from reasoning_agent.llm import GeminiClient, LLMError, MockLLM, make_llm


def test_mock_returns_responses_in_order():
    llm = MockLLM(["first", "second"])
    assert llm.generate("sys", "hello") == "first"
    assert llm.generate("sys", "again") == "second"


def test_mock_records_every_call():
    llm = MockLLM(["ok"])
    llm.generate("you are a planner", "what is 2+2?", json_mode=True)
    assert llm.calls == [{"system": "you are a planner", "user": "what is 2+2?", "json_mode": True}]


def test_mock_raises_when_out_of_responses():
    llm = MockLLM([])
    with pytest.raises(LLMError):
        llm.generate("sys", "hello")


def test_make_llm_builds_gemini_client():
    llm = make_llm(Settings(llm_provider="gemini", gemini_api_key="fake-key", gemini_model="some-model"))
    assert isinstance(llm, GeminiClient)
    assert llm.model == "some-model"


def test_make_llm_rejects_unknown_provider():
    with pytest.raises(ValueError):
        make_llm(Settings(llm_provider="not-a-provider"))


def test_gemini_client_requires_a_key():
    with pytest.raises(LLMError):
        GeminiClient(api_key="", model="gemini-3.5-flash")


# ---- retries and model fallback, with the Gemini SDK call replaced by a fake ----

def gemini_error(code, message):
    from google.genai import errors
    cls = errors.ClientError if code < 500 else errors.ServerError
    return cls(code, {"error": {"code": code, "message": message, "status": "X"}})


class FakeModels:
    """Stands in for client.models: plays back a script of errors / texts per model."""

    def __init__(self, script):
        self.script = script  # {model: [Exception or str, ...]}
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append(model)
        outcome = self.script[model].pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return type("Response", (), {"text": outcome})()


def fake_client(monkeypatch, script, models="m1,m2"):
    monkeypatch.setattr("time.sleep", lambda seconds: None)  # don't actually wait in tests
    client = GeminiClient(api_key="fake-key", model=models)
    client._client = type("Client", (), {"models": FakeModels(script)})()
    return client


def test_retries_temporary_errors_on_the_same_model(monkeypatch):
    client = fake_client(monkeypatch, {"m1": [gemini_error(503, "overloaded"), "ok"]})
    assert client.generate("s", "u") == "ok"
    assert client._client.models.calls == ["m1", "m1"]


def test_switches_model_when_daily_quota_is_used_up(monkeypatch):
    client = fake_client(monkeypatch, {
        "m1": [gemini_error(429, "Generate content requests per day limit exceeded on Free Tier")],
        "m2": ["from m2", "again m2"],
    })
    assert client.generate("s", "u") == "from m2"
    assert client.model == "m2"  # stays on the fallback model
    assert client.generate("s", "u") == "again m2"
    assert client._client.models.calls == ["m1", "m2", "m2"]


def test_switches_model_after_repeated_overload(monkeypatch):
    client = fake_client(monkeypatch, {"m1": [gemini_error(504, "deadline")] * 3, "m2": ["ok"]})
    assert client.generate("s", "u") == "ok"


def test_raises_when_all_models_fail(monkeypatch):
    daily = "requests per day limit exceeded"
    client = fake_client(monkeypatch, {"m1": [gemini_error(429, daily)], "m2": [gemini_error(429, daily)]})
    with pytest.raises(LLMError, match="All Gemini models failed"):
        client.generate("s", "u")


def test_non_retryable_error_is_raised_immediately(monkeypatch):
    client = fake_client(monkeypatch, {"m1": [gemini_error(400, "bad request")], "m2": ["unused"]})
    with pytest.raises(LLMError, match="400"):
        client.generate("s", "u")
