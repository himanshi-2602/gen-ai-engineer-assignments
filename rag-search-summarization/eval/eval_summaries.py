"""Step 3c: automatic summary evaluation with ROUGE against the human-written highlights.

For 30 test documents, compare:
- llm:    Gemini summary of the article, asked for about as many words as the reference highlights.
- lead3:  the first three sentences of the article (the standard extractive baseline for news).
Lead-3 is also scored on all 1,200 test documents (free) to show the 30-document sample is typical.

ROUGE-1 / ROUGE-2 / ROUGE-L F1 (with stemming) measure word / word-pair / longest-common-subsequence
overlap with the reference. Also reported: length adherence (actual words / requested words).

    .venv/bin/python eval/eval_summaries.py
Output: eval/results/summary_metrics.json/.md and summaries_rouge.jsonl (every summary, resumable)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rouge_score import rouge_scorer  # noqa: E402

from ragsearch.config import Settings  # noqa: E402
from ragsearch.data import load_docs  # noqa: E402
from ragsearch.llm import make_llm  # noqa: E402
from ragsearch.summarize import Summarizer, lead_n  # noqa: E402

N_DOCS = 30
OUT = ROOT / "eval/results"
LOG = OUT / "summaries_rouge.jsonl"
SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def rouge(prediction: str, reference: str) -> dict:
    scores = SCORER.score(reference, prediction)
    return {name: s.fmeasure for name, s in scores.items()}


def mean_scores(rows, key) -> dict:
    return {m: float(np.mean([r[key][m] for r in rows])) for m in ("rouge1", "rouge2", "rougeL")}


def main() -> None:
    settings = Settings.from_env()
    docs = load_docs(settings)
    testset = json.loads((ROOT / "eval/data/testset.json").read_text())
    test_docs = [q["doc_id"] for q in testset]
    sample = [q["doc_id"] for q in testset if q["type"] == "llm"][:N_DOCS]

    done = {}
    if LOG.exists():
        done = {r["doc_id"]: r for r in map(json.loads, LOG.read_text().splitlines()) if r.get("mode") == "llm"}
    summarizer = Summarizer(make_llm(settings))
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for doc_id in sample:
        reference = docs["highlights"][doc_id]
        if doc_id in done:
            rows.append(done[doc_id])
            continue
        target = int(round(len(reference.split()) / 10) * 10)
        result = summarizer.summarize([(doc_id, docs["title"][doc_id], docs["text"][doc_id])], length=target)
        lead3 = lead_n(docs["text"][doc_id], 3)
        row = {"doc_id": doc_id, "reference": reference, "llm_summary": result.summary, "lead3": lead3,
               "target_words": result.target_words, "llm_words": result.actual_words,
               "mode": result.mode, "model": result.model,
               "llm": rouge(result.summary, reference), "lead3_rouge": rouge(lead3, reference)}
        rows.append(row)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"doc {doc_id}: ROUGE-L llm {row['llm']['rougeL']:.3f} vs lead3 {row['lead3_rouge']['rougeL']:.3f} "
              f"({result.actual_words}/{target} words, {result.mode}, {result.model})", flush=True)

    llm_rows = [r for r in rows if r["mode"] == "llm"]
    all_lead3 = [{"lead3_rouge": rouge(lead_n(docs["text"][d], 3), docs["highlights"][d])} for d in test_docs]
    ratios = [r["llm_words"] / r["target_words"] for r in llm_rows]
    metrics = {
        "documents": len(llm_rows),
        "llm": mean_scores(llm_rows, "llm"),
        "lead3_same_docs": mean_scores(llm_rows, "lead3_rouge"),
        "lead3_all_test_docs": {**mean_scores(all_lead3, "lead3_rouge"), "documents": len(all_lead3)},
        "length_ratio_mean": float(np.mean(ratios)) if ratios else None,
        "length_within_20pct": float(np.mean([0.8 <= x <= 1.2 for x in ratios])) if ratios else None,
        "models": sorted({r["model"] for r in llm_rows}),
    }
    (OUT / "summary_metrics.json").write_text(json.dumps(metrics, indent=1))
    m = metrics
    md = [f"# Summary evaluation (ROUGE F1 vs. human highlights, {m['documents']} documents)", "",
          "| System | ROUGE-1 | ROUGE-2 | ROUGE-L |", "|---|---|---|---|",
          f"| Gemini summary | {m['llm']['rouge1']:.3f} | {m['llm']['rouge2']:.3f} | {m['llm']['rougeL']:.3f} |",
          f"| Lead-3 (same documents) | {m['lead3_same_docs']['rouge1']:.3f} | {m['lead3_same_docs']['rouge2']:.3f} | {m['lead3_same_docs']['rougeL']:.3f} |",
          f"| Lead-3 (all {m['lead3_all_test_docs']['documents']} test documents) | {m['lead3_all_test_docs']['rouge1']:.3f} | "
          f"{m['lead3_all_test_docs']['rouge2']:.3f} | {m['lead3_all_test_docs']['rougeL']:.3f} |", "",
          f"Length adherence: actual / requested words = {m['length_ratio_mean']:.2f} on average; "
          f"{m['length_within_20pct']:.0%} of summaries within ±20% of the requested length.", "",
          f"Models: {', '.join(m['models'])}"]
    (OUT / "summary_metrics.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
