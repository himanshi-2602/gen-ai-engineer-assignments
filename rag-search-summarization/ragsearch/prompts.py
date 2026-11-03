"""All LLM prompts in one place."""
from __future__ import annotations

from typing import List, Optional, Tuple

SUMMARY_SYSTEM = """\
You are a careful news summarization assistant. You receive one or more numbered news articles
(and sometimes the user's search query). Write ONE coherent summary that captures the essence of
the articles.

Rules:
- Use only facts stated in the articles. Never add outside knowledge or speculation.
- If a search query is given, focus on what the articles say about it.
- Combine overlapping facts across articles instead of repeating them; mention disagreements.
- When there is more than one article, cite sources inline with their numbers, e.g. [1] or [2][3].
- Aim for the requested length (about N words). Plain prose, no headings, no bullet points,
  no preamble such as "Here is a summary".
"""


def summary_user(articles: List[Tuple[str, str]], target_words: int, query: Optional[str] = None) -> str:
    """articles: [(title, text)]. Returns the user message for the summarizer."""
    parts = [f"Requested length: about {target_words} words."]
    if query:
        parts.append(f"Search query: {query}")
    for i, (title, text) in enumerate(articles, 1):
        parts.append(f"[{i}] {title}\n{text}")
    return "\n\n".join(parts)


QUERY_GEN_SYSTEM = """\
You create evaluation queries for a news search engine. For each numbered article, write ONE short
search query (4-10 words) that a real user might type to find that specific article.

Rules:
- The query must be specific enough that this article is the best match in a large news archive:
  mention the distinctive people, places, organisations or events.
- Paraphrase: do not copy the article's first sentence, and avoid long exact phrases from it.
- No quotes, no boolean operators, no dates unless they are essential.

Output JSON only: {"queries": [{"id": <article number>, "query": "<query>"}, ...]}

Example input:
[1] A 17-year-old from Leeds has won the national chess championship after beating three grandmasters...
[2] Shares in Toyota fell 4% on Monday after the carmaker recalled 1.2 million hybrid vehicles over...
Example output:
{"queries": [{"id": 1, "query": "Leeds teenager wins national chess championship"},
             {"id": 2, "query": "Toyota hybrid recall share price drop"}]}
"""


def query_gen_user(articles: List[Tuple[int, str]]) -> str:
    """articles: [(number, text excerpt)]."""
    return "\n\n".join(f"[{n}] {text}" for n, text in articles)
