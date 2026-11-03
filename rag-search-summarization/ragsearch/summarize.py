"""Summarization of retrieved documents with an LLM (plus extractive baselines / fallback)."""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

from . import prompts
from .llm import LLMClient, LLMError
from .text import split_sentences

LENGTH_PRESETS = {"short": 60, "medium": 150, "long": 300}
MIN_WORDS, MAX_WORDS = 30, 500
INPUT_WORD_BUDGET = 3500  # total article words sent to the LLM (cost / latency control)


@dataclass
class SummaryResult:
    summary: str
    target_words: int
    actual_words: int
    mode: str              # "llm" or "extractive_fallback"
    model: Optional[str]
    seconds: float
    sources: List[int]     # doc ids, in citation order [1], [2], ...

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_length(length) -> int:
    """'short' / 'medium' / 'long' or a number of words (clamped to 30–500)."""
    if isinstance(length, str) and length in LENGTH_PRESETS:
        return LENGTH_PRESETS[length]
    try:
        return max(MIN_WORDS, min(MAX_WORDS, int(length)))
    except (TypeError, ValueError):
        raise ValueError(f"length must be one of {list(LENGTH_PRESETS)} or a number of words")


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words]) + " …"


def lead_n(text: str, n: int = 3) -> str:
    """Lead-N baseline: the first n sentences (a strong baseline for news, which puts key facts first)."""
    return " ".join(split_sentences(text)[:n])


def extractive_summary(texts: List[str], target_words: int) -> str:
    """Fallback without an LLM: lead sentences from each document, round-robin, up to the length."""
    sentences = [split_sentences(t) for t in texts]
    out, count, i = [], 0, 0
    while count < target_words and any(i < len(s) for s in sentences):
        for doc_number, s in enumerate(sentences, 1):
            if i < len(s) and count < target_words:
                out.append(s[i] + (f" [{doc_number}]" if len(texts) > 1 else ""))
                count += len(s[i].split())
        i += 1
    return " ".join(out)


class Summarizer:
    def __init__(self, llm: Optional[LLMClient]):
        self.llm = llm

    def summarize(self, documents: List[Tuple[int, str, str]], length="medium",
                  query: Optional[str] = None) -> SummaryResult:
        """documents: [(doc_id, title, text)] in the order they should be cited."""
        if not documents:
            raise ValueError("no documents to summarize")
        target = resolve_length(length)
        per_doc = max(150, INPUT_WORD_BUDGET // len(documents))
        articles = [(title, truncate_words(text, per_doc)) for _, title, text in documents]
        start = time.time()
        try:
            if self.llm is None:
                raise LLMError("no LLM configured")
            summary = self.llm.generate(prompts.SUMMARY_SYSTEM, prompts.summary_user(articles, target, query))
            summary = re.sub(r"\s+", " ", summary).strip()
            mode, model = "llm", getattr(self.llm, "model", None)
        except LLMError:
            # Degrade gracefully: the user still gets a (simpler) summary.
            summary = extractive_summary([text for _, _, text in documents], target)
            mode, model = "extractive_fallback", None
        return SummaryResult(summary=summary, target_words=target, actual_words=len(summary.split()),
                             mode=mode, model=model, seconds=round(time.time() - start, 2),
                             sources=[doc_id for doc_id, _, _ in documents])
