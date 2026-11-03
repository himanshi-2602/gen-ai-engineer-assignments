# AI-judge evaluation of the 10 end-to-end summaries

**Rater: an AI model (Claude Opus 5.5) acting as judge, not a human.** It used the same rubric as the human
rating sheet ([`eval/human/to_rate.md`](../human/to_rate.md)): 1 = poor, 5 = excellent; length_ok = 1/0.
To make the faithfulness scores evidence-based rather than impressionistic, every specific, checkable claim
in each summary (names, numbers, quotes, places) was looked up in the **full text** of the article it cites.

| Criterion | Mean (1–5) | Min | Max |
|---|---|---|---|
| Coherence | **4.4** | 3 | 5 |
| Coverage | **4.7** | 4 | 5 |
| Faithfulness | **4.6** | 3 | 5 |
| Relevance to the query | **4.5** | 2 | 5 |
| Length appropriate | **10/10** | | |

| # | Query | Words (target) | Coh. | Cov. | Faith. | Rel. | Notes |
|---|---|---|---|---|---|---|---|
| 1 | Teenagers adopting privacy measures and vague-booking on social media | 57 (60) | 5 | 4 | 4 | 5 | Accurate core (un-tagging, parallel identities, vague-booking). "Private accounts … [1][2][3]" is over-cited: only [1] mentions private accounts. |
| 2 | Russia confirms North Korean leader Kim Jong Un Moscow visit | 146 (150) | 5 | 5 | 5 | 5 | Captures the confirmation **and** the later cancellation across 5 articles; all facts verified (executions, 90% debt write-off, "Friendship Year"). The Ukraine point is cited [2][3] but appears only in [3]. |
| 3 | KFC apologizes after broadcasting sex scene during family dinner | 302 (300) | 3 | 4 | 5 | **2** | The incident is summarized accurately (Outlander, 75,000 views, apology, channel blocking), but **~60% of the text covers unrelated KFC stories** (food stylist, chicken farms, a fight) from off-topic results, to fill 300 words. |
| 4 | Disabled girl banned from rides at Blackpool Pleasure Beach | 59 (60) | 5 | 5 | 5 | 5 | Focused on the one relevant article; correctly ignores the unrelated Blackpool stories. |
| 5 | George Ford inspires Bath comeback against Leinster | 148 (150) | 5 | 5 | 4 | 5 | Accurate match narrative (15-5 at half-time, solo try, 18-15 loss, Madigan penalties). Some quotes ("pocket general", "dazzling break", Garvey) are cited [3][5] but are only in [3]. |
| 6 | Watford fan Nick Cruwys attacked after Wolves match | 327 (300) | 4 | 5 | 5 | 5 | Thorough and accurate across 5 articles (arrests, bail, £1,000 reward, £25,000 fund, club statements); slightly list-like; 9% over length. |
| 7 | Pennsylvania bride shoots niece dead wedding night | 61 (60) | 5 | 5 | 5 | 5 | Concise: event, competing accounts (deliberate vs accidental), verdict. Unrelated shooting stories correctly ignored. |
| 8 | Body of missing student Jennifer Houle found in Mississippi River | 165 (150) | 5 | 5 | 5 | 5 | All details verified (drowning, bridge height 60–110 ft, purse and phone, Pi Beta Phi, Carlson School); 10% over length. |
| 9 | lion gets head stuck in feeding barrel at zoo | 304 (300) | 3 | 5 | 5 | **3** | The lion incident is accurate and complete, but **the second half covers unrelated zoo stories** (giraffe death, lion-encounter cage, buffalo) from off-topic results. |
| 10 | Kenyan police arrest suspects after Garissa University College attack | 61 (60) | 4 | 4 | **3** | 5 | **One unsupported phrase:** the slow response was "defended as a complex tactical operation", but the source says it was defended with a football **goalkeeper analogy**. Uses `[1, 3, 4]` instead of the requested `[1][3][4]` citation style. |

## Findings
1. **Faithfulness is high:** across about 70 verified claims, one unsupported characterization (#10) and a few over-broad citations (#1, #2, #5). No invented events, people or numbers.
2. **Length control works:** all 10 within ±10% of the request (60, 150 or 300 words).
3. **Main weakness: padding with off-topic sources at long lengths.** When the user asks for 300 words but only one or two of the selected articles match the query (#3, #9), the summarizer fills the length with the other, unrelated articles. At 60 words (#4, #7) it correctly ignores them. Fixes: (a) tell the summarizer to skip sources unrelated to the query even if the summary ends up shorter; (b) only pass documents whose re-ranker score clears a threshold; (c) show the user a "low relevance" marker on such results.

## Limitation
An AI judge is not a substitute for human evaluation: it may share blind spots with the summarizing model and has
no reader's perspective. These scores are an interim, evidence-checked assessment; human ratings can be added in
[`eval/human/ratings.csv`](../human/ratings.csv) and aggregated with `python eval/human_eval.py report`.
