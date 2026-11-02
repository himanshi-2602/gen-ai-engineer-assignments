"""API tests: the real FastAPI app, with the agent replaced by one using a MockLLM."""
import pytest
from fastapi.testclient import TestClient

import api
from helpers import good_attempt
from reasoning_agent.agent import ReasoningAgent
from reasoning_agent.llm import MockLLM


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")  # make sure no real LLM is ever called
    with TestClient(api.app) as test_client:
        yield test_client
    api.app.dependency_overrides.clear()


def use_mock_agent(replies):
    api.app.dependency_overrides[api.get_agent] = lambda: ReasoningAgent(MockLLM(replies), max_retries=0)


def test_solve_returns_agent_json(client):
    use_mock_agent(good_attempt())
    response = client.post("/solve", json={"question": "Train 14:30 to 18:05?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "3 hours 35 minutes"
    assert body["status"] == "success"
    assert body["metadata"]["retries"] == 0


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "x" * 1001}, {"q": "hi"}])
def test_solve_validates_input(client, payload):
    use_mock_agent(good_attempt())
    assert client.post("/solve", json=payload).status_code == 422


def test_without_api_key_service_reports_not_configured(client):
    assert client.get("/health").json() == {"status": "not_configured"}
    response = client.post("/solve", json={"question": "1+1?"})
    assert response.status_code == 503
    assert "GEMINI_API_KEY" in response.json()["detail"]


def test_index_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "fetch(\"/solve\"" in response.text
