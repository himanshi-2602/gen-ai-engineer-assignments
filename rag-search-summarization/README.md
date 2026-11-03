# Document Search and Summarization with LLMs

Search about 25,000 CNN/DailyMail news articles with **hybrid retrieval** (BM25 keywords + bge-small sentence embeddings, fused with Reciprocal Rank Fusion, then re-ranked by a cross-encoder). Then **summarize** the results with Gemini at a length you choose. Includes a web interface with auto-suggest, pagination and a summary-length slider, plus a full evaluation: retrieval accuracy for 4 methods, ROUGE against human-written highlights, and human ratings.

📄 **The detailed report is in [REPORT.md](REPORT.md):** data pre-processing and its justification, methodology, evaluation procedure and results, challenges and solutions, scalability.

---

## Problem

Design a system that can **search and summarize large amounts of text** using an LLM:
1. **Data preparation:** choose a sizable corpus; pre-process and clean it for search and summarization.
2. **Document search:** take user queries and return the **top N** relevant documents or excerpts, combining **traditional IR methods with LLM embeddings**.
3. **Document summarization:** summarize the retrieved documents with an LLM, coherently, capturing their essence, at a **user-specified length**.
4. **Evaluation:** build a test set of documents with a generated query each; measure **search accuracy**; evaluate summaries with **automated metrics (ROUGE) and human evaluation**.
5. **Interface (bonus):** a user-friendly UI with **auto-suggestion**, **pagination** and **adjustable summary lengths**.

Deliverables: a detailed report, the full codebase with a README, and (bonus) hosting or local setup instructions. Graded on functionality, accuracy, scalability, efficiency, report quality and code quality.

---

## Requirements checklist

| Requirement | Where |
|---|---|
| Sizable corpus, pre-processed and cleaned | 24,841 CNN/DailyMail articles → 114,983 passages. [`ragsearch/data.py`](ragsearch/data.py), [`ragsearch/text.py`](ragsearch/text.py); [REPORT §2](REPORT.md#2-data-preparation) |
| Search: traditional IR + LLM embeddings | BM25 + bge-small embeddings (FAISS), RRF fusion, cross-encoder re-ranking. [`ragsearch/search.py`](ragsearch/search.py) |
| Top N documents or excerpts | `/api/search` returns ranked documents with their best-matching excerpt, paginated |
| LLM summarization, coherent, user-chosen length | Gemini with citations [1][2]; `short` / `medium` / `long` or 30–500 words. [`ragsearch/summarize.py`](ragsearch/summarize.py) |
| Test set: a query generated per test document | 200 LLM-written queries + 1,000 highlight queries. [`eval/make_testset.py`](eval/make_testset.py) → [`eval/data/testset.json`](eval/data/testset.json) |
| Search accuracy | Recall@1/5/10, MRR@10, nDCG@10 and latency for 4 methods. [`eval/eval_search.py`](eval/eval_search.py) |
| ROUGE | Gemini vs Lead-3 baseline against human highlights. [`eval/eval_summaries.py`](eval/eval_summaries.py) |
| Human evaluation | Rating sheet and report. [`eval/human_eval.py`](eval/human_eval.py) → [`eval/human/`](eval/human/) |
| UI with auto-suggest, pagination, adjustable length (bonus) | [`app.py`](app.py) + [`static/`](static/) |
| Report: preprocessing, methodology, evaluation, challenges | [REPORT.md](REPORT.md) |
| README with setup and run instructions; local setup (bonus) | This file |

---

## Setup

Requirements: Python 3.9+, about 1.5 GB of disk (Python packages and models), and internet access for the first run.

```bash
cd rag-search-summarization
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

cp .env.example .env      # then put your Gemini API key in .env (free key: https://aistudio.google.com/apikey)
```
The API key is only needed for **summaries** and for generating evaluation queries. **Search works without it**, and summaries fall back to an extractive method if the LLM is unavailable. No key is included in this repository: `.env` is git-ignored.

## Run

```bash
# 1. Download + clean + chunk the corpus (~65 MB download, ~20 s)
.venv/bin/python scripts/prepare_data.py

# 2. Build the BM25 + embedding indexes and the auto-suggest list
#    (~10 min on an Apple M-series GPU; slower on CPU; the models are downloaded once, ~230 MB)
.venv/bin/python scripts/build_index.py

# 3. Start the web app
.venv/bin/uvicorn app:app --port 8002
#    UI:       http://localhost:8002
#    API docs: http://localhost:8002/docs
```

**Using the UI:** type a query; suggestions appear as you type (use the arrow keys and Enter, or click). Browse the results with **Prev / Next / page numbers**. The top 5 results are pre-selected for the summary; tick or untick any results, including ones on other pages. Set the length with the **slider** (30–500 words) or the **Short / Medium / Long** buttons, then click **Summarize**. Citations like **[1]** link to the source articles. Click any title to read the full article.

**API examples:**
```bash
curl "localhost:8002/api/search?q=ebola+outbreak+west+africa&method=hybrid_rerank&page=1&page_size=10"
curl "localhost:8002/api/suggest?q=manch"
curl -X POST localhost:8002/api/summarize -H "Content-Type: application/json" \
     -d '{"query": "ebola outbreak west africa", "top_k": 5, "length": "short"}'
```

## Evaluation

```bash
.venv/bin/python eval/make_testset.py     # test queries (20 LLM calls); the committed testset.json is reused
.venv/bin/python eval/eval_search.py      # search metrics → eval/results/search_metrics.md
.venv/bin/python eval/eval_summaries.py   # ROUGE → eval/results/summary_metrics.md (30 LLM calls, resumable)
.venv/bin/python eval/human_eval.py prepare   # → eval/human/to_rate.md + ratings.csv to fill in
.venv/bin/python eval/human_eval.py report    # → eval/results/human_eval.md
```
Or run everything with `./run_all.sh`. The results are summarized in [REPORT.md §5](REPORT.md#5-evaluation).

## Tests
```bash
.venv/bin/python -m pytest -q     # offline unit tests (no models, indexes or API key needed)
```

## Configuration (`.env`)
| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | (none) | Needed for LLM summaries and query generation |
| `GEMINI_MODEL` | 4 Gemini Flash models | Comma-separated **fallback list** (the free tier has small per-model daily quotas) |
| `LLM_MIN_INTERVAL` | `6` | Seconds between LLM calls (free-tier rate limits) |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Sentence-embedding model for dense retrieval |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder for re-ranking |
| `DATA_DIR` | `./data` | Where the corpus and indexes are stored (git-ignored) |

To index the **full** CNN/DailyMail corpus (~310,000 articles) instead of the ~25,000-article subset: `python scripts/prepare_data.py --splits train validation test`, then rebuild the index. See [REPORT §6](REPORT.md#6-scalability-and-efficiency) for scaling notes.

## Project structure
```
rag-search-summarization/
├── ragsearch/
│   ├── config.py      # Settings (.env)
│   ├── text.py        # Cleaning, sentence splitting, chunking
│   ├── data.py        # Download, clean, de-duplicate, chunk → Parquet
│   ├── index.py       # BM25 index + embeddings (FAISS); fp16 encoder
│   ├── search.py      # BM25 / dense / hybrid (RRF) / hybrid + cross-encoder re-rank; pagination
│   ├── summarize.py   # LLM summaries with length control + citations; Lead-3; extractive fallback
│   ├── suggest.py     # Auto-suggest from frequent named phrases
│   ├── prompts.py     # All LLM prompts
│   └── llm.py         # Gemini client (retries, timeouts, model fallback) + MockLLM
├── app.py             # FastAPI: /api/search, /api/suggest, /api/summarize, /api/document, UI
├── static/            # Web UI (HTML/CSS/vanilla JS)
├── scripts/           # prepare_data.py, build_index.py
├── eval/              # make_testset.py, eval_search.py, eval_summaries.py, human_eval.py
│   ├── data/          # testset.json (committed)
│   ├── results/       # metrics (committed)
│   └── human/         # rating sheet + ratings
├── tests/             # offline unit tests
├── REPORT.md          # detailed report
└── requirements.txt, .env.example, run_all.sh
```
