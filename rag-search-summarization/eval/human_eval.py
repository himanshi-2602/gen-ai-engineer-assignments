"""Step 3d: human evaluation of multi-document summaries (the end-to-end search -> summary flow).

    .venv/bin/python eval/human_eval.py prepare   # creates eval/human/to_rate.md and ratings.csv
    # ... a person reads to_rate.md and fills the 1-5 scores in ratings.csv ...
    .venv/bin/python eval/human_eval.py report    # averages -> eval/results/human_eval.md

`prepare` takes 10 test queries, retrieves the top 5 documents (hybrid + re-rank) and summarizes
them at different requested lengths (60 / 150 / 300 words), so length control is checked as well.

Rating criteria (1 = poor, 5 = excellent):
- coherence:    reads as one well-organised, fluent text.
- coverage:     captures the essence / main points of the retrieved articles.
- faithfulness: everything stated is supported by the articles (no invented facts).
- relevance:    focuses on what the query asks about.
- length_ok:    1 if the length matches the request reasonably, else 0.
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HUMAN = ROOT / "eval/human"
CRITERIA = ("coherence", "coverage", "faithfulness", "relevance")
LENGTHS = (60, 150, 300)


def prepare() -> None:
    from ragsearch.config import Settings
    from ragsearch.llm import make_llm
    from ragsearch.search import SearchEngine
    from ragsearch.summarize import Summarizer, truncate_words

    settings = Settings.from_env()
    engine = SearchEngine(settings)
    summarizer = Summarizer(make_llm(settings))
    testset = [q for q in json.loads((ROOT / "eval/data/testset.json").read_text()) if q["type"] == "llm"]
    queries = random.Random(7).sample(testset, 10)

    HUMAN.mkdir(parents=True, exist_ok=True)
    md = ["# Summaries to rate", "",
          "For each item, read the query, the source excerpts and the summary, then fill in `ratings.csv`",
          "with scores from 1 (poor) to 5 (excellent): coherence, coverage, faithfulness, relevance;",
          "and length_ok = 1 if the length roughly matches the request, else 0.", ""]
    rows = []
    for i, item in enumerate(queries, 1):
        hits = engine.search(item["query"], "hybrid_rerank", 1, 5).results
        documents = [(h.doc_id, h.title, engine.docs["text"][h.doc_id]) for h in hits]
        target = LENGTHS[(i - 1) % len(LENGTHS)]
        result = summarizer.summarize(documents, target, item["query"])
        md += [f"## {i}. Query: {item['query']}", "",
               f"Requested length: {target} words · actual: {result.actual_words} words · {result.model}", "",
               "**Sources (first 120 words of each):**", ""]
        md += [f"{n}. *{h.title}*: {truncate_words(engine.docs['text'][h.doc_id], 120)}" for n, h in enumerate(hits, 1)]
        md += ["", "**Summary:**", "", f"> {result.summary}", ""]
        rows.append({"id": i, "query": item["query"], "target_words": target, "actual_words": result.actual_words,
                     "model": result.model or result.mode, **{c: "" for c in CRITERIA}, "length_ok": "", "comment": ""})
        print(f"{i}. {item['query']} -> {result.actual_words}/{target} words", flush=True)

    (HUMAN / "to_rate.md").write_text("\n".join(md))
    with (HUMAN / "ratings.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Written {HUMAN / 'to_rate.md'} and {HUMAN / 'ratings.csv'} (fill in the scores)")


def report() -> None:
    with (HUMAN / "ratings.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if all(r[c].strip() for c in CRITERIA)]
    if not rows:
        raise SystemExit("No completed ratings in eval/human/ratings.csv yet.")
    lines = [f"# Human evaluation ({len(rows)} multi-document summaries, 1–5 scale)", "",
             "| Criterion | Mean | Min | Max |", "|---|---|---|---|"]
    for c in CRITERIA:
        values = [float(r[c]) for r in rows]
        lines.append(f"| {c} | {sum(values) / len(values):.2f} | {min(values):.0f} | {max(values):.0f} |")
    length_ok = [int(r["length_ok"]) for r in rows if r["length_ok"].strip()]
    if length_ok:
        lines += ["", f"Length judged appropriate: {sum(length_ok)}/{len(length_ok)}"]
    ratios = [int(r["actual_words"]) / int(r["target_words"]) for r in rows]
    lines += [f"Actual / requested words: {sum(ratios) / len(ratios):.2f} on average", "",
              "| # | Query | Words (target) | Coh. | Cov. | Faith. | Rel. | Comment |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['id']} | {r['query']} | {r['actual_words']} ({r['target_words']}) | {r['coherence']} | "
                     f"{r['coverage']} | {r['faithfulness']} | {r['relevance']} | {r['comment']} |")
    out = ROOT / "eval/results/human_eval.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    {"prepare": prepare, "report": report}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: print(__doc__))()
