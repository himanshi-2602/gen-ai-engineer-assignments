"""Web API and user interface.

    .venv/bin/uvicorn app:app --port 8002
    open http://localhost:8002        (search UI)    http://localhost:8002/docs   (API docs)
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Literal, Optional, Union

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ragsearch.config import Settings
from ragsearch.llm import LLMError, make_llm
from ragsearch.search import SearchEngine
from ragsearch.suggest import Suggester
from ragsearch.summarize import Summarizer

log = logging.getLogger("uvicorn.error")
STATIC = Path(__file__).resolve().parent / "static"
DEFAULT_METHOD = "hybrid_rerank"
Method = Literal["bm25", "dense", "hybrid", "hybrid_rerank"]  # validated: anything else -> 422


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load indexes and models once at startup; every request reuses them.
    settings = Settings.from_env()
    app.state.engine = SearchEngine(settings)
    app.state.suggester = Suggester.load(settings.index_dir / "suggestions.json")
    try:
        llm = make_llm(settings)
    except LLMError as err:
        log.warning("No LLM configured (%s): summaries will use the extractive fallback.", err)
        llm = None
    app.state.summarizer = Summarizer(llm)
    log.info("Search engine ready in %.1fs", app.state.engine.load_seconds)
    yield


app = FastAPI(title="Document search and summarization",
              description="Hybrid BM25 + embedding search with re-ranking, and LLM summaries of the results.",
              version="1.0", lifespan=lifespan)


class SummarizeRequest(BaseModel):
    query: Optional[str] = Field(None, max_length=300, description="Search query (used to retrieve and focus)")
    doc_ids: Optional[List[int]] = Field(None, max_length=10, description="Documents to summarize; "
                                         "if omitted, the top `top_k` search results for `query` are used")
    top_k: int = Field(5, ge=1, le=10)
    length: Union[int, str] = Field("medium", description="'short' (~60 words), 'medium' (~150), "
                                    "'long' (~300) or a number of words (30–500)")
    method: Method = DEFAULT_METHOD


@app.get("/api/health")
def health(request: Request) -> dict:
    engine: SearchEngine = request.app.state.engine
    return {"status": "ok", "documents": len(engine.docs["doc_id"]), "chunks": len(engine.chunk_text),
            "llm": request.app.state.summarizer.llm is not None}


@app.get("/api/search")
def search(request: Request, q: str = Query(..., min_length=1, max_length=300),
           method: Method = Query(DEFAULT_METHOD),
           page: int = Query(1, ge=1, le=100), page_size: int = Query(10, ge=1, le=50)) -> dict:
    return request.app.state.engine.search(q.strip(), method, page, page_size).to_dict()


@app.get("/api/suggest")
def suggest(request: Request, q: str = Query("", max_length=100), limit: int = Query(8, ge=1, le=20)) -> dict:
    return {"query": q, "suggestions": request.app.state.suggester.suggest(q, limit)}


@app.get("/api/document/{doc_id}")
def document(request: Request, doc_id: int) -> dict:
    engine: SearchEngine = request.app.state.engine
    if not 0 <= doc_id < len(engine.docs["doc_id"]):
        raise HTTPException(status_code=404, detail="document not found")
    return engine.document(doc_id)


@app.post("/api/summarize")
def summarize(request: Request, body: SummarizeRequest) -> dict:
    engine: SearchEngine = request.app.state.engine
    if body.doc_ids:
        doc_ids = body.doc_ids
    elif body.query:
        doc_ids = [r.doc_id for r in engine.search(body.query, body.method, 1, body.top_k).results]
    else:
        raise HTTPException(status_code=422, detail="give a query or doc_ids")
    if any(not 0 <= d < len(engine.docs["doc_id"]) for d in doc_ids):
        raise HTTPException(status_code=404, detail="document not found")
    if not doc_ids:
        raise HTTPException(status_code=404, detail="no documents matched the query")
    documents = [(d, engine.docs["title"][d], engine.docs["text"][d]) for d in doc_ids]
    try:
        result = request.app.state.summarizer.summarize(documents, body.length, body.query)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err))
    return {**result.to_dict(), "source_titles": [engine.docs["title"][d] for d in doc_ids]}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
