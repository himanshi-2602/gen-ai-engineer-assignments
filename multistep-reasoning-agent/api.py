"""HTTP API.

    uvicorn api:app --reload --port 8000
    curl -X POST localhost:8000/solve -H "Content-Type: application/json" \
         -d '{"question": "Alice has 3 red apples and twice as many green apples as red. How many apples?"}'

Interactive docs: http://localhost:8000/docs
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse

from reasoning_agent.agent import ReasoningAgent, build_agent
from reasoning_agent.llm import LLMError
from reasoning_agent.schemas import SolveRequest, SolveResponse

log = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the agent once at startup and reuse it for every request.
    try:
        app.state.agent = build_agent()
    except LLMError as err:
        # Start anyway so /health explains the problem instead of the server crashing.
        log.error("Agent not configured: %s", err)
        app.state.agent, app.state.config_error = None, str(err)
    yield


app = FastAPI(
    title="Multi-step reasoning agent",
    description="Planner -> executor -> verifier agent with self-checking and retries.",
    version="1.0",
    lifespan=lifespan,
)


def get_agent(request: Request) -> ReasoningAgent:
    """Dependency: hands the shared agent to endpoints (tests replace it with a mock)."""
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(status_code=503,
                            detail=f"Agent is not configured: {getattr(request.app.state, 'config_error', '')}")
    return agent


@app.get("/health")
def health(request: Request) -> dict:
    ready = getattr(request.app.state, "agent", None) is not None
    return {"status": "ok" if ready else "not_configured"}


@app.post("/solve", response_model=SolveResponse)
def solve(body: SolveRequest, agent: ReasoningAgent = Depends(get_agent)) -> SolveResponse:
    # A plain `def` (not async): FastAPI runs it in a worker thread, so slow LLM calls
    # don't block other requests.
    return agent.solve(body.question.strip())


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Minimal page for trying the agent in a browser (it calls POST /solve)."""
    return INDEX_HTML


INDEX_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reasoning Agent</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 720px; margin: 32px auto; padding: 0 16px; color: #1c2430; }
  textarea { width: 100%; box-sizing: border-box; font: inherit; padding: 10px; border: 1px solid #c9d1dc; border-radius: 8px; }
  button { margin-top: 8px; padding: 9px 18px; border: 0; border-radius: 8px; background: #0a84ff; color: #fff; font-weight: 600; cursor: pointer; }
  button:disabled { opacity: .6; cursor: wait; }
  .chips button { background: #eef2f7; color: #1c2430; font-weight: 400; padding: 5px 10px; margin: 4px 4px 0 0; }
  .card { border: 1px solid #d9dfe7; border-radius: 10px; padding: 16px; margin-top: 18px; }
  .status { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 13px; color: #fff; }
  .success { background: #1f9d55; } .failed { background: #c62828; }
  .muted { color: #5d6b7e; font-size: 14px; }
  pre { background: #f4f6f9; padding: 12px; border-radius: 8px; overflow-x: auto; font-size: 13px; }
</style></head>
<body>
  <h1>Multi-step reasoning agent</h1>
  <p class="muted">Planner &rarr; executor &rarr; verifier, with self-checking and retries. Each question takes about 30 seconds.</p>
  <textarea id="q" rows="3">A train leaves at 22:40 and arrives at 01:15 the next day. How long is the journey?</textarea>
  <div class="chips">
    <button type="button">Alice has 3 red apples and twice as many green apples as red. How many apples does she have in total?</button>
    <button type="button">A meeting needs 60 minutes. There are free slots: 09:00–09:30, 09:45–10:30, 11:00–12:00. Which slots can fit the meeting?</button>
    <button type="button">Alice has 3 apples and gives away 5 of them. How many apples does she have left?</button>
  </div>
  <button id="go">Solve</button>
  <div id="out"></div>
<script>
  const q = document.getElementById("q"), go = document.getElementById("go"), out = document.getElementById("out");
  document.querySelectorAll(".chips button").forEach(b => b.onclick = () => { q.value = b.textContent; });
  const esc = s => String(s).replace(/[&<>]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;"}[c]));
  go.onclick = async () => {
    go.disabled = true; out.innerHTML = '<p class="muted">Planning, executing and verifying...</p>';
    try {
      const r = await fetch("/solve", { method: "POST", headers: { "Content-Type": "application/json" },
                                        body: JSON.stringify({ question: q.value }) });
      const d = await r.json();
      if (!r.ok) { out.innerHTML = '<div class="card">Error ' + r.status + ': ' + esc(JSON.stringify(d.detail)) + '</div>'; return; }
      const checks = d.metadata.checks.map(c => (c.passed ? "✅ " : "❌ ") + c.check_name + " — " + c.details).join("\\n");
      out.innerHTML = '<div class="card"><span class="status ' + d.status + '">' + d.status + '</span>' +
        '<h2>' + esc(d.answer) + '</h2><p>' + esc(d.reasoning_visible_to_user) + '</p>' +
        '<details><summary class="muted">Debug: plan, checks (' + d.metadata.checks.length + ') and retries (' +
        d.metadata.retries + ')</summary><p class="muted">' + esc(d.metadata.plan) + '</p><pre>' + esc(checks) +
        '</pre><pre>' + esc(JSON.stringify(d, null, 2)) + '</pre></details></div>';
    } catch (e) { out.innerHTML = '<div class="card">Request failed: ' + esc(e) + '</div>'; }
    finally { go.disabled = false; }
  };
</script>
</body></html>"""
