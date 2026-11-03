# Summary evaluation (ROUGE F1 vs. human highlights, 30 documents)

| System | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| Gemini summary | 0.363 | 0.107 | 0.233 |
| Lead-3 (same documents) | 0.395 | 0.166 | 0.237 |
| Lead-3 (all 1200 test documents) | 0.406 | 0.180 | 0.255 |

Length adherence: actual / requested words = 1.00 on average; 100% of summaries within ±20% of the requested length.

Models: gemini-3-flash-preview, gemini-3.1-flash-lite, gemini-3.5-flash
