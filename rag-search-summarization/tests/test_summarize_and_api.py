import pytest
from fastapi.testclient import TestClient

from ragsearch import prompts
from ragsearch.llm import MockLLM
from ragsearch.search import DocResult, SearchResponse
from ragsearch.suggest import Suggester
from ragsearch.summarize import Summarizer, extractive_summary, lead_n, resolve_length, truncate_words

DOCS = [(0, "Kidney chain", "Zully gave a kidney. Six patients were helped. A computer matched them."),
        (1, "Data match", "A new program finds donor chains. It saved lives. Doctors praised it.")]


def test_resolve_length():
    assert resolve_length("short") == 60 and resolve_length("long") == 300
    assert resolve_length(5) == 30 and resolve_length(9999) == 500 and resolve_length("120") == 120
    with pytest.raises(ValueError):
        resolve_length("huge")


def test_llm_summary_uses_prompt_with_length_query_and_numbered_sources():
    llm = MockLLM(["  A computer matched donors [1][2].  "])
    result = Summarizer(llm).summarize(DOCS, length=80, query="kidney donation")
    assert result.summary == "A computer matched donors [1][2]."
    assert result.mode == "llm" and result.sources == [0, 1] and result.target_words == 80
    user = llm.calls[0]["user"]
    assert "about 80 words" in user and "kidney donation" in user
    assert "[1] Kidney chain" in user and "[2] Data match" in user
    assert llm.calls[0]["system"] == prompts.SUMMARY_SYSTEM


def test_falls_back_to_extractive_summary_when_llm_fails():
    result = Summarizer(MockLLM([])).summarize(DOCS, length=30)
    assert result.mode == "extractive_fallback"
    assert result.summary.startswith("Zully gave a kidney. [1] A new program finds donor chains. [2]")


def test_baselines_and_truncation():
    assert lead_n("One. Two. Three. Four.", 3) == "One. Two. Three."
    assert extractive_summary(["One two three. Four five six."], 3) == "One two three."
    assert truncate_words("a b c d", 2) == "a b …"


def test_prompts_specify_format_and_examples():
    assert "cite sources inline" in prompts.SUMMARY_SYSTEM
    assert "Output JSON only" in prompts.QUERY_GEN_SYSTEM and "Example" in prompts.QUERY_GEN_SYSTEM


# ---------- API, with a fake search engine (no models or indexes needed) ----------

class FakeEngine:
    docs = {"doc_id": [0, 1], "title": ["Kidney chain", "Data match"], "text": [d[2] for d in DOCS],
            "words": [12, 12]}
    chunk_text = ["chunk a", "chunk b"]

    def search(self, query, method, page, page_size):
        results = [DocResult(doc_id=d, rank=i + 1, score=1.0, title=self.docs["title"][d], excerpt="...",
                             chunk_id=d, words=12, matched_by=["bm25"]) for i, d in enumerate([0, 1])]
        return SearchResponse(query, method, 2, page, page_size, results[:page_size], {"total": 1.0})

    def document(self, doc_id):
        return {"doc_id": doc_id, "title": self.docs["title"][doc_id], "text": self.docs["text"][doc_id]}


@pytest.fixture
def client():
    import app as app_module

    app = app_module.app
    app.router.lifespan_context = lambda _app: _NoLifespan()  # skip loading real models in tests
    app.state.engine = FakeEngine()
    app.state.suggester = Suggester([["David Cameron", 9]])
    app.state.summarizer = Summarizer(MockLLM(["Summary [1]."] * 5))
    with TestClient(app) as test_client:
        yield test_client


class _NoLifespan:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *args):
        return False


def test_api_search_suggest_document_and_health(client):
    body = client.get("/api/search", params={"q": "kidney", "page_size": 1}).json()
    assert body["total"] == 2 and len(body["results"]) == 1 and body["results"][0]["title"] == "Kidney chain"
    assert client.get("/api/suggest", params={"q": "cam"}).json()["suggestions"] == ["David Cameron"]
    assert client.get("/api/document/1").json()["title"] == "Data match"
    assert client.get("/api/document/99").status_code == 404
    assert client.get("/api/health").json()["documents"] == 2


def test_api_summarize_by_query_and_by_ids(client):
    by_query = client.post("/api/summarize", json={"query": "kidney", "top_k": 2, "length": "short"}).json()
    assert by_query["summary"] == "Summary [1]." and by_query["sources"] == [0, 1] and by_query["target_words"] == 60
    by_ids = client.post("/api/summarize", json={"doc_ids": [1], "length": 120}).json()
    assert by_ids["source_titles"] == ["Data match"]


def test_api_validates_input(client):
    assert client.get("/api/search", params={"q": ""}).status_code == 422
    assert client.get("/api/search", params={"q": "x", "method": "magic"}).status_code == 422
    assert client.post("/api/summarize", json={}).status_code == 422
    assert client.post("/api/summarize", json={"doc_ids": [5]}).status_code == 404
    assert client.post("/api/summarize", json={"doc_ids": [0], "length": "huge"}).status_code == 422


def test_ui_is_served(client):
    assert "News Search" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
