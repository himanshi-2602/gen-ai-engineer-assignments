"""Step 3a: build the search test set (queries whose correct answer is a known document).

Two query sets, both "known-item" (the source document is the relevant result):
- llm:       for 200 random documents, Gemini writes a short paraphrased query a user might type
             (batched 10 articles per call -> 20 LLM calls). This is the set the assignment asks for.
- highlight: for 1,000 other random documents, the first human-written highlight sentence is used as
             the query. Free (no LLM) and larger, but closer in wording to the article.

    .venv/bin/python eval/make_testset.py
Output: eval/data/testset.json (committed, so the evaluation is reproducible without an API key).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ragsearch import prompts  # noqa: E402
from ragsearch.config import Settings  # noqa: E402
from ragsearch.data import load_docs  # noqa: E402
from ragsearch.llm import make_llm  # noqa: E402
from ragsearch.summarize import truncate_words  # noqa: E402
from ragsearch.text import split_sentences  # noqa: E402

SEED, LLM_QUERIES, HIGHLIGHT_QUERIES, BATCH = 42, 200, 1000, 10
OUT = ROOT / "eval/data/testset.json"


def main() -> None:
    settings = Settings.from_env()
    docs = load_docs(settings)
    rng = random.Random(SEED)
    sample = rng.sample(range(len(docs["doc_id"])), LLM_QUERIES + HIGHLIGHT_QUERIES)
    llm_docs, highlight_docs = sample[:LLM_QUERIES], sample[LLM_QUERIES:]

    existing = json.loads(OUT.read_text()) if OUT.exists() else []
    done = {(q["doc_id"], q["type"]) for q in existing}
    items = list(existing)

    for doc_id in highlight_docs:
        if (doc_id, "highlight") not in done:
            items.append({"doc_id": doc_id, "type": "highlight",
                          "query": split_sentences(docs["highlights"][doc_id])[0].rstrip(".")})

    llm = make_llm(settings)
    todo = [d for d in llm_docs if (d, "llm") not in done]
    for start in range(0, len(todo), BATCH):
        batch = todo[start:start + BATCH]
        articles = [(i + 1, truncate_words(docs["text"][d], 250)) for i, d in enumerate(batch)]
        reply = json.loads(llm.generate(prompts.QUERY_GEN_SYSTEM, prompts.query_gen_user(articles), json_mode=True))
        by_number = {int(q["id"]): q["query"].strip() for q in reply["queries"]}
        for i, d in enumerate(batch, 1):
            if by_number.get(i):
                items.append({"doc_id": d, "type": "llm", "query": by_number[i], "model": llm.model})
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(items, indent=1, ensure_ascii=False))  # save progress after every batch
        print(f"LLM queries: {sum(q['type'] == 'llm' for q in items)}/{LLM_QUERIES} ({llm.model})", flush=True)

    OUT.write_text(json.dumps(items, indent=1, ensure_ascii=False))
    print(f"Test set: {sum(q['type'] == 'llm' for q in items)} LLM queries, "
          f"{sum(q['type'] == 'highlight' for q in items)} highlight queries -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
