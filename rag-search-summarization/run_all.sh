#!/usr/bin/env bash
# Full pipeline: data -> indexes -> evaluation. Requires GEMINI_API_KEY in .env for the LLM steps.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
$PY scripts/prepare_data.py          # 1. download + clean + chunk (~20 s)
$PY scripts/build_index.py           # 2. BM25 + embeddings + auto-suggest (~10 min on Apple M-series GPU)
$PY eval/make_testset.py             # 3a. test queries (20 LLM calls; reuses eval/data/testset.json if present)
$PY eval/eval_search.py              # 3b. search accuracy + latency for 4 methods
$PY eval/eval_summaries.py           # 3c. ROUGE vs highlights (30 LLM calls; resumable)
$PY eval/human_eval.py prepare       # 3d. summaries for human rating (10 LLM calls)
