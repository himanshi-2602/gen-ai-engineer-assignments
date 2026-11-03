# Paired bootstrap: MRR@10 differences (95% CI, 10,000 resamples)

## llm queries (n = 200)

| A − B | Δ MRR | 95% CI | Significant? |
|---|---|---|---|
| hybrid_rerank − bm25 | +0.044 | [+0.017, +0.073] | yes |
| hybrid_rerank − dense | +0.019 | [-0.016, +0.055] | no |
| hybrid_rerank − hybrid | +0.013 | [-0.015, +0.041] | no |
| hybrid − bm25 | +0.031 | [+0.015, +0.049] | yes |
| hybrid − dense | +0.006 | [-0.023, +0.037] | no |
| dense − bm25 | +0.025 | [-0.011, +0.061] | no |

## highlight queries (n = 1000)

| A − B | Δ MRR | 95% CI | Significant? |
|---|---|---|---|
| hybrid_rerank − bm25 | +0.014 | [+0.002, +0.026] | yes |
| hybrid_rerank − dense | +0.086 | [+0.068, +0.103] | yes |
| hybrid_rerank − hybrid | +0.032 | [+0.019, +0.046] | yes |
| hybrid − bm25 | -0.018 | [-0.029, -0.008] | yes |
| hybrid − dense | +0.053 | [+0.040, +0.066] | yes |
| dense − bm25 | -0.071 | [-0.089, -0.054] | yes |
