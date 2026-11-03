"""Step 3b: measure search accuracy and latency for every method on the test set.

For each query the relevant document is the one it was written for (known-item search). Metrics:
- Recall@k (k = 1, 5, 10): share of queries whose document is in the top k.
- MRR@10: mean of 1/rank of the document (0 if not in the top 10).
- nDCG@10: with one relevant document, 1/log2(rank + 1) (0 if not in the top 10).
- Latency: mean and 95th percentile per query.

    .venv/bin/python eval/eval_search.py
Output: eval/results/search_metrics.json and search_metrics.md
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ragsearch.config import Settings  # noqa: E402
from ragsearch.search import METHODS, SearchEngine  # noqa: E402

OUT = ROOT / "eval/results"


def evaluate(engine: SearchEngine, queries, method: str) -> dict:
    ranks, latencies = [], []
    for item in queries:
        start = time.perf_counter()
        ranking, _ = engine.rank(item["query"], method)
        latencies.append((time.perf_counter() - start) * 1000)
        ids = [doc for doc, *_ in ranking]
        ranks.append(ids.index(item["doc_id"]) + 1 if item["doc_id"] in ids else None)

    def recall(k):
        return sum(1 for r in ranks if r and r <= k) / len(ranks)

    return {
        "queries": len(ranks),
        "recall@1": recall(1), "recall@5": recall(5), "recall@10": recall(10),
        "mrr@10": sum(1 / r for r in ranks if r and r <= 10) / len(ranks),
        "ndcg@10": sum(1 / math.log2(r + 1) for r in ranks if r and r <= 10) / len(ranks),
        "not_found_in_top100": sum(r is None for r in ranks) / len(ranks),
        "latency_ms_mean": float(np.mean(latencies)), "latency_ms_p95": float(np.percentile(latencies, 95)),
        "ranks": ranks,
    }


def main() -> None:
    settings = Settings.from_env()
    engine = SearchEngine(settings)
    testset = json.loads((ROOT / "eval/data/testset.json").read_text())
    for warmup in ("warm up", "model loading"):
        for method in METHODS:
            engine.rank(warmup, method)

    results = {}
    for qtype in ("llm", "highlight"):
        queries = [q for q in testset if q["type"] == qtype]
        for method in METHODS:
            metrics = evaluate(engine, queries, method)
            results[f"{qtype}/{method}"] = metrics
            print(f"{qtype:9} {method:14} R@1 {metrics['recall@1']:.3f}  R@10 {metrics['recall@10']:.3f}  "
                  f"MRR {metrics['mrr@10']:.3f}  {metrics['latency_ms_mean']:.0f} ms", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "search_metrics.json").write_text(json.dumps(results, indent=1))
    lines = [f"# Search evaluation ({len(engine.docs['doc_id']):,} documents, {len(engine.chunk_text):,} chunks)", ""]
    for qtype, title in (("llm", "LLM-generated queries"), ("highlight", "Highlight queries")):
        n = results[f"{qtype}/bm25"]["queries"]
        lines += [f"## {title} ({n} queries)", "",
                  "| Method | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Mean latency | p95 latency |",
                  "|---|---|---|---|---|---|---|---|"]
        for method in METHODS:
            m = results[f"{qtype}/{method}"]
            lines.append(f"| {method} | {m['recall@1']:.3f} | {m['recall@5']:.3f} | {m['recall@10']:.3f} | "
                         f"{m['mrr@10']:.3f} | {m['ndcg@10']:.3f} | {m['latency_ms_mean']:.0f} ms | "
                         f"{m['latency_ms_p95']:.0f} ms |")
        lines.append("")
    (OUT / "search_metrics.md").write_text("\n".join(lines))
    print(f"Written {OUT / 'search_metrics.md'}")


if __name__ == "__main__":
    main()
