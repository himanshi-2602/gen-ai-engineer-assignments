# Search evaluation (24,841 documents, 114,968 chunks)

## LLM-generated queries (200 queries)

| Method | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Mean latency | p95 latency |
|---|---|---|---|---|---|---|---|
| bm25 | 0.800 | 0.955 | 0.970 | 0.862 | 0.889 | 1 ms | 1 ms |
| dense | 0.830 | 0.960 | 0.990 | 0.887 | 0.912 | 15 ms | 13 ms |
| hybrid | 0.840 | 0.965 | 0.985 | 0.893 | 0.916 | 11 ms | 13 ms |
| hybrid_rerank | 0.855 | 0.970 | 0.985 | 0.906 | 0.926 | 188 ms | 241 ms |

## Highlight queries (1000 queries)

| Method | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Mean latency | p95 latency |
|---|---|---|---|---|---|---|---|
| bm25 | 0.805 | 0.931 | 0.952 | 0.860 | 0.883 | 1 ms | 1 ms |
| dense | 0.708 | 0.897 | 0.923 | 0.788 | 0.822 | 13 ms | 12 ms |
| hybrid | 0.780 | 0.929 | 0.944 | 0.842 | 0.867 | 11 ms | 13 ms |
| hybrid_rerank | 0.825 | 0.930 | 0.962 | 0.874 | 0.895 | 226 ms | 280 ms |
