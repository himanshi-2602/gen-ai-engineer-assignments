# Report: Document Search and Summarization Using LLMs

**System:** hybrid retrieval (BM25 + bge-small embeddings, Reciprocal Rank Fusion, cross-encoder re-ranking) over 24,841 CNN/DailyMail news articles, with Gemini summaries of the retrieved documents at a user-chosen length, served through a FastAPI web app.

**Headline results**

| | Result |
|---|---|
| Search accuracy, LLM-generated queries (200) | **MRR@10 0.906, Recall@1 85.5%, Recall@10 98.5%** (hybrid + re-rank) |
| Search accuracy, highlight queries (1,000) | **MRR@10 0.874, Recall@1 82.5%, Recall@10 96.2%** (hybrid + re-rank) |
| Best method | Hybrid + re-rank: **significantly better than BM25 on both query sets** (paired bootstrap) and never significantly worse than any alternative |
| Search latency | Hybrid ~11 ms; hybrid + re-rank ~190–230 ms per query (Apple M5) |
| Summary length control | Actual/requested words = **1.00** on average; **100%** of summaries within ±20% |
| ROUGE vs. human highlights | Gemini R-1/R-2/R-L = 0.363 / 0.107 / 0.233; Lead-3 baseline 0.395 / 0.166 / 0.237 (discussed in §5.3) |
| Summary quality (AI judge, 10 end-to-end summaries) | Coherence 4.4, coverage 4.7, **faithfulness 4.6**, relevance 4.5 (1–5); length OK 10/10. Main weakness: padding with off-topic sources at 300 words (§5.4) |

---

## 1. Overview and design choices

```
            ┌─────────── offline (once) ───────────┐            ┌──────────── online (per query) ─────────────┐
 CNN/DM ──► clean ─► de-dup ─► chunk (≈200 words) ─┬─► BM25 index (stemmed, stop-words removed)
 parquet                                           └─► bge-small embeddings ─► FAISS (cosine)
                                                                                 query ─► BM25 top-200 chunks ─┐
                                                                                       ─► dense top-200 chunks ┴─► best chunk per document
                                                                                          ─► Reciprocal Rank Fusion ─► cross-encoder re-rank (top 30)
                                                                                          ─► top-N documents + excerpts (paginated)
                                                                                          ─► Gemini summary of the selected documents (length N words, citations)
```

| Decision | Choice | Alternatives considered | Why |
|---|---|---|---|
| Corpus | **CNN/DailyMail 3.0.0**, validation + test splits (24,858 articles, 16.9M words) | Wikipedia, arXiv abstracts, BBC News (2,225 docs) | Sizable and real, and it comes with **human-written highlights**, the reference summaries that ROUGE needs. Only a 65 MB download; the code also supports the full 310k-article corpus (`--splits train validation test`). |
| Retrieval unit | **Passages of ~200 words** (whole sentences, ~50-word overlap), aggregated to documents | Whole documents | Embedding models truncate long inputs (bge: 512 tokens) and a whole-article vector blurs topics. Passages give precise matching **and** the excerpt to show (the assignment's "documents or excerpts"). |
| Traditional IR | **BM25** (bm25s, sparse matrices, Snowball stemming, stop-words removed) | TF-IDF | BM25 is the standard strong lexical baseline (term-frequency saturation, length normalisation). bm25s scores 115k passages in about 1 ms. |
| LLM embeddings | **BAAI/bge-small-en-v1.5** (33M parameters, 384-d), local | OpenAI or Gemini embedding APIs; all-MiniLM-L6-v2 | Strong retrieval quality for its size, **free, offline and with no API quota**. API embeddings would need hundreds of rate-limited calls to index and one call per query. |
| Vector index | **FAISS** `IndexFlatIP` (exact cosine) | HNSW / IVF-PQ | Exact search is ~10 ms at this size. FAISS lets the same code switch to approximate indexes for larger corpora (§6). |
| Fusion | **Reciprocal Rank Fusion** (k = 60) | Weighted score sum | BM25 scores and cosine similarities have different scales; RRF uses only ranks, so it needs no calibration or tuning. |
| Re-ranking | **cross-encoder/ms-marco-MiniLM-L-6-v2** on the top 30 | None; an LLM re-ranker | A cross-encoder reads query and passage together (more accurate than either retriever) and is cheap on a short list; an LLM re-ranker would cost an API call per search. |
| Summarizer | **Gemini Flash** (fallback list of 4 models) | Local seq2seq (BART/PEGASUS); fine-tuning | Strong zero-shot multi-document summaries with precise length control and citations. See §3.3 on fine-tuning. |
| Serving | **FastAPI** + vanilla-JS UI | Streamlit, Gradio | A real HTTP API that other clients can use, plus a lightweight UI with auto-suggest, pagination and length control. |

---

## 2. Data preparation

### 2.1 What the raw data looked like
Measured on the 24,858 raw articles before designing any cleaning rule:

| Observation | Count | Impact if left alone |
|---|---|---|
| Source tags at the start: `(CNN)`, `London (CNN) --`, `(WIRED)`, `(The Hollywood Reporter)`, `(Travel + Leisure)`… | 2,298 articles (9%) | Pollutes titles, BM25 terms and summaries |
| Tokenizer artefact: a space before punctuation (`sentence . Next`) | 22,804 articles (**92%**) | Breaks sentence splitting and looks wrong in excerpts and summaries |
| Boilerplate sentences: "Scroll down for video" (6,514), "Click here…" (275), "READ MORE" (85), "All rights reserved", "Follow @…" | 6,997 sentences in 6,886 articles (35,598 words) | Noise in search and summaries |
| Exact or near duplicates | 2 | Duplicate results |
| Stubs under 80 words (captions, video pages) | 15 | Useless results, poor chunks |
| No titles in the dataset | all | Results need a readable heading |

### 2.2 Pipeline (`ragsearch/text.py`, `ragsearch/data.py`)
1. **Unicode NFKC normalisation** and whitespace collapsing: unifies curly quotes, odd spaces and similar.
2. **Remove the source tag** at the start: an optional dateline plus a short capitalised name in brackets. Ordinary brackets elsewhere are kept (unit-tested).
3. **Fix spaces before punctuation** (`end . Next` becomes `end. Next`).
4. **Sentence splitting** (rule-based, tuned for news): splits after `. ! ?`, keeps closing quotes and brackets with their sentence, and does **not** split after abbreviations (Mr., Dr., U.S., a.m., St., …) or initials (J. K. Rowling).
5. **Drop boilerplate sentences** (click here, read more, follow us, scroll down for video, copyright…).
6. **Drop stubs** of fewer than 80 words, and **de-duplicate** with a fingerprint of the lower-cased letters only, which also catches copies that differ in spacing or punctuation.
7. **Chunk** into passages of whole sentences of up to ~200 words, with consecutive passages sharing ~50 words, so a fact on a boundary is fully inside at least one passage.
8. **Pseudo-title** = the first sentence (shortened to 18 words), for display.
9. **Reference summaries:** highlight lines joined into sentences (used by ROUGE).
10. Saved as **Parquet** (`docs.parquet`, `chunks.parquet`): columnar, compressed and fast to load. The same format and code work for the full corpus.

**Known limitation:** in 44 cases the source text has no space after a full stop (`defence.CLICK HERE to see more`), so a boilerplate phrase is merged with real content into one long "sentence", and the filter drops both: 2,560 words, **0.015% of the corpus**. A fix would remove only the boilerplate clause, or insert the missing space first.

**Deliberately not done:** lower-casing, stemming or stop-word removal on the stored text. Those are applied only inside the BM25 tokenizer; embeddings and the LLM work best on natural text.

### 2.3 Result
| | Value |
|---|---|
| Raw articles | 24,858 (13,368 validation + 11,490 test) |
| Removed | 2 duplicates, 15 stubs |
| **Documents** | **24,841** (16.7M words; median 603 words, 95th percentile 1,365) |
| **Passages** | **114,968** (4.6 per document; median 186 words, 95th percentile 199) |
| Reference highlights | mean 52.8 words |
| Preparation time | ~19 s |

---

## 3. Methodology

### 3.1 Search (`ragsearch/search.py`)
For a query, each retriever returns its top 200 **passages**. A document's rank in a retriever is the rank of its best passage, and that passage becomes the **excerpt** shown to the user.
- **BM25:** the query is tokenized exactly like the corpus (stemmed, stop words removed). A query made only of stop words returns no BM25 results instead of an error.
- **Dense:** the query is prefixed with bge's retrieval instruction ("Represent this sentence for searching relevant passages: "), embedded and L2-normalised; FAISS returns the passages with the highest cosine similarity.
- **Hybrid (RRF):** `score(d) = Σ_m 1 / (60 + rank_m(d))` over the two retrievers. Documents found by both rank higher (unit-tested). Each result records which retriever found it, shown as "keyword match" or "semantic match" in the UI.
- **Hybrid + re-rank (default):** the cross-encoder re-scores the top 30 fused documents using their best passage.
- **Top N and pagination:** the ranking is kept to 100 documents and served page by page (`page`, `page_size`).

### 3.2 Summarization (`ragsearch/summarize.py`, `ragsearch/prompts.py`)
- **Input:** the selected documents (by default the top 5 results), numbered [1]…[n], with the query. Each article is truncated to share a **3,500-word input budget**, which bounds cost and latency regardless of how many documents are selected.
- **Prompt rules:** use only facts from the articles; focus on the query; merge overlapping facts and mention disagreements; **cite sources inline as [1][2]**; aim for N words; plain prose with no preamble.
- **Length control:** `short` (60), `medium` (150), `long` (300) or any number from 30 to 500 words. The UI has a slider plus presets.
- **Graceful degradation:** if the LLM fails (outage or quota), the API returns an **extractive summary** (lead sentences of each document, round-robin, with citations) and labels it as such, instead of an error.
- **Reliability:** the Gemini client has a 60-second timeout, retries with backoff on 429/5xx errors, and a **model fallback list**: when a model's free daily quota is used up, the next model is used.

### 3.3 Why not fine-tune the LLM? (assignment hint)
I considered fine-tuning and decided against it:
- **Retrieval already supplies the domain knowledge.** The summarizer reads the actual articles at query time, so fine-tuning wouldn't teach it facts it needs.
- **Cost and resources:** fine-tuning a capable LLM needs paid GPU time or a paid tuning API, and the assignment itself notes that LLMs are resource-intensive.
- **Measured need:** length control is already almost perfect (§5.3), and the summary ratings (§5.4) show the zero-shot summaries are coherent and faithful; their main weakness (padding with off-topic sources) is a prompt and selection issue, not something fine-tuning is needed for.

Fine-tuning would make sense for a strict house style, or for a much smaller self-hosted model, e.g. a distilled summarizer trained on the 287k CNN/DM training pairs, to cut per-summary cost at scale.

### 3.4 Auto-suggest (`ragsearch/suggest.py`)
Suggestions are **named phrases mined from the corpus**: runs of 1–4 capitalised words, cleaned of function words, weekdays and months, and ranked by how many documents mention them (19,682 phrases appearing in at least 5 documents). Typing a prefix matches the start of **any word** in a phrase ("cam" gives *David Cameron*, *Cameron*, *Cambridge*; "world c" gives *World Cup*, *Cricket World Cup*), using binary search over sorted keys. Suggestions therefore always lead to something that exists in the archive.

---

## 4. Interface (bonus)
`uvicorn app:app --port 8002` serves:
- **UI** at `/`: a search box with **auto-suggest** (debounced, keyboard navigation); a method selector; results with highlighted query terms, a ~60-word snippet, the score and match badges; **pagination** (Prev / pages / Next, 5/10/20 per page); **shareable URLs** (`/?q=...&page=2`); tick-boxes to choose which results to summarize (top 5 pre-selected, kept across pages); a **length slider (30–500 words) and Short/Medium/Long presets**; summaries with clickable **[n] citations**; a full-article reader.
- **REST API:** `GET /api/search`, `GET /api/suggest`, `POST /api/summarize`, `GET /api/document/{id}`, `GET /api/health`, with OpenAPI docs at `/docs`. Inputs are validated (an invalid method or length gives 422; an unknown document gives 404).

Hosting: the server needs about 1.6 GB of RAM (embedding model, cross-encoder and indexes), which is more than common free hosting tiers (~512 MB). So the deliverable is the **detailed local setup** in the README. A hosted version would need a ~2 GB instance, or the models moved to an inference service.

---

## 5. Evaluation

### 5.1 Test set (`eval/make_testset.py` → `eval/data/testset.json`)
As the assignment specifies, a subset of the corpus is the test set, and **each test document gets a query that should retrieve it** (known-item search: the source document is the relevant result).

| Query set | Size | How the query was made | Example |
|---|---|---|---|
| **LLM queries** | 200 random documents | Gemini writes a short, **paraphrased** query a user might type, specific enough to identify the article (prompt in `prompts.py`; 10 articles per call, so only 20 API calls) | "Prince William criticized over elephant show visit in China" |
| **Highlight queries** | 1,000 other random documents | The first human-written highlight sentence (no LLM cost; more queries, so tighter confidence intervals) | "Jonathan Trott is recalled to the England squad for their West Indies tour" |

The test set is committed, so the search evaluation is reproducible without an API key.

### 5.2 Search accuracy (`eval/eval_search.py`, `eval/significance.py`)
Metrics: **Recall@k** (is the document in the top k?), **MRR@10** (mean of 1/rank), **nDCG@10**, and latency per query.

**LLM-generated queries (200)**

| Method | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Mean latency |
|---|---|---|---|---|---|---|
| BM25 | 0.800 | 0.955 | 0.970 | 0.862 | 0.889 | 1 ms |
| Embeddings (dense) | 0.830 | 0.960 | **0.990** | 0.887 | 0.912 | 15 ms |
| Hybrid (RRF) | 0.840 | 0.965 | 0.985 | 0.893 | 0.916 | 11 ms |
| **Hybrid + re-rank** | **0.855** | **0.970** | 0.985 | **0.906** | **0.926** | 188 ms |

**Highlight queries (1,000)**

| Method | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Mean latency |
|---|---|---|---|---|---|---|
| BM25 | 0.805 | **0.931** | 0.952 | 0.860 | 0.883 | 1 ms |
| Embeddings (dense) | 0.708 | 0.897 | 0.923 | 0.788 | 0.822 | 13 ms |
| Hybrid (RRF) | 0.780 | 0.929 | 0.944 | 0.842 | 0.867 | 11 ms |
| **Hybrid + re-rank** | **0.825** | 0.930 | **0.962** | **0.874** | **0.895** | 226 ms |

**Statistical significance:** paired bootstrap 95% confidence intervals for MRR@10 differences, from 10,000 resamples ([`search_significance.md`](eval/results/search_significance.md)).

| Comparison | LLM queries: Δ MRR [95% CI] | Highlight queries: Δ MRR [95% CI] |
|---|---|---|
| hybrid + re-rank − BM25 | **+0.044 [+0.017, +0.073]** ✔ | **+0.014 [+0.002, +0.026]** ✔ |
| hybrid + re-rank − dense | +0.019 [−0.016, +0.055] | **+0.086 [+0.068, +0.103]** ✔ |
| hybrid + re-rank − hybrid | +0.013 [−0.015, +0.041] | **+0.032 [+0.019, +0.046]** ✔ |
| hybrid − BM25 | **+0.031 [+0.015, +0.049]** ✔ | **−0.018 [−0.029, −0.008]** ✔ (worse) |
| dense − BM25 | +0.025 [−0.011, +0.061] | **−0.071 [−0.089, −0.054]** ✔ (worse) |

**Analysis**
- **Hybrid + re-rank is the best method overall.** It has the highest Recall@1, MRR and nDCG on both query sets, is significantly better than BM25 on both, and is **never significantly worse than any alternative**. That's why it's the default.
- **Which single retriever wins depends on the query style.** With paraphrased queries (LLM set), embeddings beat BM25: they match meaning, not exact words. With highlight queries, which share much of their wording with the article, BM25 is much stronger than embeddings (−0.071 MRR for dense). A real system sees both kinds of query, so neither retriever alone is safe. That's the main argument for combining them.
- **Plain RRF fusion has a cost:** equal-weight RRF is pulled down by the weaker retriever on highlight queries (−0.018 vs BM25). The **cross-encoder re-ranker fixes this** (+0.032 over hybrid) because it judges the query–passage pair directly.
- **Recall@10 is 95–99%** for all methods, so the correct article is almost always on the first page. The methods differ mainly in placing it **first**.
- **Caveat (conservative metric):** only the source article counts as relevant, but CNN/DM contains several articles about the same story, so a "miss" may still be a relevant result. The true relevance of the results is therefore at least as high as reported.

### 5.3 Summary quality: ROUGE (`eval/eval_summaries.py`)
For 30 test documents, Gemini summarized each article with a requested length equal to the reference highlights' length (rounded to 10 words). Scores are ROUGE F1 with stemming against the **human-written highlights**.

| System | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| Gemini summary (30 docs) | 0.363 | 0.107 | 0.233 |
| Lead-3 baseline (same 30 docs) | 0.395 | 0.166 | 0.237 |
| Lead-3 baseline (all 1,200 test docs) | 0.406 | 0.180 | 0.255 |

**Length adherence:** actual/requested words = **1.00** on average; **100%** of summaries within ±20% of the requested length. Across the 10 multi-document summaries for human evaluation (60/150/300 words), all were within ±10% of the requested length.

**Interpretation, stated honestly:**
- **Lead-3 (the article's first three sentences) scores higher on ROUGE-2 and slightly higher on ROUGE-1; ROUGE-L is about equal.** This is a well-known property of CNN/DailyMail: news puts the key facts first, and the highlights were largely written from the opening paragraphs, so a baseline that *copies* the opening shares many exact word pairs with them. Lead-3 is famously hard to beat on this dataset, even for trained summarization models.
- ROUGE measures **word overlap**, not quality. The Gemini summaries **paraphrase and condense** (fewer exact bigrams, hence lower ROUGE-2), while Lead-3 is not a summary at all: it can't be focused on a query, can't combine several articles, and can't follow a requested length. Published human studies of LLM news summaries find the same pattern: zero-shot GPT-3 summaries scored lower ROUGE than fine-tuned models on CNN/DM, yet human readers preferred them (Goyal, Li & Durrett, 2022, *News Summarization and Evaluation in the Era of GPT-3*). That's why the human evaluation below matters.
- The 30-document Lead-3 scores (0.395/0.166/0.237) are close to the 1,200-document scores (0.406/0.180/0.255), so the 30-document sample is representative.
- Because of free-tier quotas, the 30 summaries were produced by three Gemini Flash variants through the fallback list (`gemini-3.5-flash`, `gemini-3-flash-preview`, `gemini-3.1-flash-lite`); every summary records its model in [`summaries_rouge.jsonl`](eval/results/summaries_rouge.jsonl).

### 5.4 Human evaluation (`eval/human_eval.py`)
Ten end-to-end cases: a test query, then the top-5 documents (hybrid + re-rank), then a Gemini summary at a requested length that cycles through 60, 150 and 300 words. The rater sees the query, the opening of each source and the summary ([`eval/human/to_rate.md`](eval/human/to_rate.md)), and scores 1–5 for **coherence**, **coverage** (captures the essence), **faithfulness** (no unsupported claims) and **relevance** to the query, plus whether the length is appropriate ([`ratings.csv`](eval/human/ratings.csv)).

**Interim result: AI-judge evaluation (an AI model, not a human), with claim checking.** The ten summaries were rated against the same rubric by an AI judge (Claude Opus 5.5). To ground the faithfulness scores, every checkable claim (names, numbers, quotes) was looked up in the full text of the cited article. Details per summary: [`eval/results/ai_judge_eval.md`](eval/results/ai_judge_eval.md).

| Criterion (1–5) | Coherence | Coverage | Faithfulness | Relevance | Length appropriate |
|---|---|---|---|---|---|
| Mean (min–max) | 4.4 (3–5) | 4.7 (4–5) | 4.6 (3–5) | 4.5 (2–5) | 10/10 |

- **Faithfulness:** out of about 70 verified claims, one unsupported characterization ("complex tactical operation", where the source uses a goalkeeper analogy) and a few over-broad citations (a fact cited to [3][5] that appears only in [3]). No invented events, people or numbers.
- **Length control:** all ten within ±10% of the requested 60/150/300 words.
- **Main weakness found:** at **300 words**, when only one or two of the five selected articles match the query, the summarizer **pads with the unrelated articles** (KFC: relevance 2/5; lion: 3/5). At 60 words it correctly ignores them. Fixes are listed in §8.
- **Human evaluation:** independent human ratings were not collected for this submission. The rating sheet ([`to_rate.md`](eval/human/to_rate.md), [`ratings.csv`](eval/human/ratings.csv)) and the aggregation script are ready, so human scores can be added with `python eval/human_eval.py report`. An AI judge is a useful, evidence-checked proxy, but it is not a substitute: it may share blind spots with the summarizing model.

---

## 6. Scalability and efficiency

**Measured on an Apple M5 Mac (24 GB RAM), corpus of 24,841 documents / 114,968 passages:**

| Stage | Time / size |
|---|---|
| Data preparation | 19 s |
| BM25 index | 6.7 s; 83 MB |
| Embeddings (bge-small, fp16 on GPU) | 573 s (~200 passages/s); FAISS index 184 MB |
| Server start-up (load indexes and models) | ~35 s; ~1.6 GB RAM |
| Query: BM25 / dense / hybrid / hybrid + re-rank | ~1 ms / ~13 ms / ~11 ms / ~190–230 ms |
| Summary (Gemini Flash, 3–5 documents) | ~2–6 s (dominated by the LLM API) |
| Auto-suggest | < 1 ms (binary search over 19,682 phrases) |

**Efficiency choices:**
- **fp16 embeddings:** 2.5× faster indexing than fp32 with identical vectors (mean cosine similarity 1.0002 against fp32 on 2,000 passages); see §7.
- **Expensive work only on short lists:** the cross-encoder scores only 30 documents, and the LLM only reads the selected documents within a 3,500-word budget.
- **One LLM call per summary**, and batched query generation (10 articles per call: 20 calls instead of 200).
- Models and indexes load **once** at server start-up.

**Scaling to the full corpus (~310k articles, ~1.45M passages), and beyond:**
- **Ingestion** is a linear batch job (Parquet in and out). Embedding is the bottleneck: about 2 hours for 1.45M passages on one laptop GPU, minutes on a cloud GPU, and it's done once (new documents are embedded incrementally).
- **BM25** (sparse matrices) grows linearly in size; query time stays in milliseconds.
- **Dense search:** exact FAISS search is O(N): ~11 ms at 115k, ~140 ms expected at 1.45M. For larger corpora, switch to FAISS **HNSW** (sub-10 ms, more memory) or **IVF-PQ** (vectors compressed from 1,536 to about 64 bytes, so 1.45M passages fit in ~100 MB). Only the index construction in `index.py` changes; the search code doesn't.
- **Beyond one machine:** shard the BM25 and vector indexes, or move them to a search service such as Elasticsearch/OpenSearch (BM25 plus kNN) or a vector database. RRF fusion and re-ranking remain the same.
- **Re-ranking and summarization cost don't grow with corpus size**: they always process a fixed short list.

---

## 7. Challenges and solutions

| # | Challenge | What happened | Solution |
|---|---|---|---|
| 1 | **Indexing seemed to hang** | The first embedding run showed no progress after 20 minutes. A stack sample showed it was working, just slowly: fp32 on Apple MPS manages ~100 passages/s, so ~19 min. | Benchmarked options on 2,000 passages: **fp16 gives 274 passages/s (2.5×) with identical vectors** (cosine 1.0002); capping inputs at 256 tokens gives 414/s but alters long passages (cosine 0.998). Chose fp16 at full length. Indexing now takes ~9.5 min, with progress logged. |
| 2 | **Sentence splitter dropped closing quotes** | A unit test showed `He said "Yes."` was cut to `He said "Yes.` The quote was consumed by the split pattern. **66% of articles (16,351) were affected.** | Fixed the pattern (a closing quote stays with its sentence), re-processed the corpus and rebuilt the indexes. Document IDs were verified unchanged, so the test set stayed valid. |
| 3 | **Splits after abbreviations** | A result title read "A year ago, when Dr." | The splitter now re-joins pieces ending in common abbreviations or initials (unit-tested); 214 titles were corrected, and the index rebuilt. |
| 4 | **Many source-tag formats** | The first rule handled `(CNN)` but missed 26 multi-word tags such as `(The Hollywood Reporter)`. | A general pattern (optional dateline plus a short capitalised name in brackets), checked to leave ordinary brackets alone. |
| 5 | **Noisy auto-suggestions** | The most frequent capitalised words were weekdays, months and "However". | A stop-list of non-topic words, and stripping them from phrase edges. |
| 6 | **LLM free-tier limits and model changes** | `gemini-2.5-flash` is retired for new keys; the newest model returned 503/504 errors; the free tier allows ~20 requests per model per day. | Model fallback list, timeouts and backoff; batched query generation (10× fewer calls); resumable evaluation scripts; an extractive fallback in the app so users still get a summary. |
| 7 | **The API accepted invalid parameters** | `method=magic` was accepted, because FastAPI's `enum=` only documents allowed values. | `Literal[...]` types, so FastAPI rejects invalid values with 422 (covered by a test). |
| 8 | **Equal-weight fusion hurt on lexical queries** | On highlight queries, hybrid RRF scored below BM25 alone. | Added the cross-encoder re-ranker, which recovers and surpasses both (§5.2). Weighted or adaptive fusion is future work, tuned on a held-out set rather than the test set. |
| 9 | **ROUGE favours extractive baselines** | Lead-3 outscores the LLM on ROUGE-2. | Reported honestly, with the reason; added length-adherence metrics and human evaluation to judge quality properly. |

---

## 8. Limitations and future work
- **Relevance labels:** known-item evaluation counts only the source article as relevant. Graded human relevance judgements for a sample of queries would measure the precision of the *other* results too.
- **Fusion weighting:** learn per-query weights (e.g. more BM25 weight for long, quote-like queries) on a separate development set.
- **Off-topic padding in long summaries (found in §5.4):** instruct the summarizer to skip sources unrelated to the query even if the summary is shorter, and pass only documents whose re-ranker score clears a threshold.
- **Human evaluation:** collect independent human ratings with the prepared sheet (several raters, with inter-rater agreement), to complement the AI-judge scores.
- **Faithfulness checking:** automatically verify that each summary sentence is supported by a cited passage (an NLI model or LLM-as-judge), and flag unsupported claims in the UI.
- **Hosting:** containerise and deploy on a ~2 GB instance; cache frequent queries and summaries.
- **Freshness:** incremental ingestion of new articles (append to BM25 and FAISS without a full rebuild).
