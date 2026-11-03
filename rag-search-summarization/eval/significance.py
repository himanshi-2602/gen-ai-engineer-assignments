"""Paired bootstrap confidence intervals for MRR@10 differences between search methods.

Resamples the queries with replacement (10,000 times) and reports the 95% interval of
MRR(method A) - MRR(method B). If the interval excludes 0, the difference is unlikely to be chance.

    .venv/bin/python eval/significance.py      (needs eval/results/search_metrics.json)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PAIRS = [("hybrid_rerank", "bm25"), ("hybrid_rerank", "dense"), ("hybrid_rerank", "hybrid"),
         ("hybrid", "bm25"), ("hybrid", "dense"), ("dense", "bm25")]


def reciprocal_ranks(ranks):
    return np.array([1 / r if r and r <= 10 else 0.0 for r in ranks])


def main() -> None:
    results = json.loads((ROOT / "eval/results/search_metrics.json").read_text())
    rng = np.random.default_rng(0)
    lines = ["# Paired bootstrap: MRR@10 differences (95% CI, 10,000 resamples)", ""]
    for qtype in ("llm", "highlight"):
        rr = {m: reciprocal_ranks(results[f"{qtype}/{m}"]["ranks"]) for m in
              ("bm25", "dense", "hybrid", "hybrid_rerank")}
        n = len(rr["bm25"])
        idx = rng.integers(0, n, size=(10_000, n))
        lines += [f"## {qtype} queries (n = {n})", "", "| A − B | Δ MRR | 95% CI | Significant? |", "|---|---|---|---|"]
        for a, b in PAIRS:
            diff = rr[a] - rr[b]
            boot = diff[idx].mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            lines.append(f"| {a} − {b} | {diff.mean():+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
                         f"{'yes' if lo > 0 or hi < 0 else 'no'} |")
        lines.append("")
    out = ROOT / "eval/results/search_significance.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
