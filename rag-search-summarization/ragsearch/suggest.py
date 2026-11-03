"""Query auto-suggestion from frequent named phrases in the corpus.

Phrases are runs of 1–4 capitalised words (people, places, organisations, events) such as
"Manchester United" or "Islamic State", ranked by how many documents mention them. Typing a prefix
matches the start of any word in the phrase ("cam" -> "David Cameron"), so suggestions only point to
things that exist in the archive.
"""
from __future__ import annotations

import bisect
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List

_PHRASE = re.compile(r"\b(?:[A-Z][a-z]+|[A-Z]{2,5})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,5})){0,3}\b")
_SKIP_FIRST = {"The", "A", "An", "But", "And", "In", "On", "At", "It", "He", "She", "They", "We", "I",
               "This", "That", "There", "His", "Her", "Their", "When", "After", "Mr", "Mrs", "Ms", "Dr"}
# Capitalised words that are not useful search topics on their own.
_NOT_TOPICS = {
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August", "September",
    "October", "November", "December", "However", "Meanwhile", "Although", "While", "Yesterday",
    "Today", "Tonight", "Last", "Now", "Then", "Some", "Many", "More", "Most", "Other", "Others",
    "According", "Speaking", "During", "Before", "Since", "Despite", "Instead", "Also", "If", "So",
    "What", "Why", "How", "Who", "Where", "Here", "These", "Those", "One", "Two", "Three", "CNN",
    "With", "Another", "People", "Every", "Even", "Only", "Just", "Each", "Both", "Our", "Your",
    "Its", "All", "Any", "No", "Not", "As", "For", "From", "By", "Of", "To", "Is", "Was", "Are",
}


def extract_phrases(text: str) -> set:
    phrases = set()
    for match in _PHRASE.finditer(text):
        words = match.group(0).split()
        while words and (words[0] in _SKIP_FIRST or words[0] in _NOT_TOPICS):
            words = words[1:]
        while words and words[-1] in _NOT_TOPICS:
            words = words[:-1]
        if words and (len(words) > 1 or len(words[0]) > 3):
            phrases.add(" ".join(words))
    return phrases


def build_suggestions(texts: Iterable[str], min_docs: int = 5, limit: int = 20000) -> List[List]:
    """[[phrase, document_count], ...] for the most frequent phrases."""
    counts = Counter()
    for text in texts:
        counts.update(extract_phrases(text))
    return [[p, n] for p, n in counts.most_common(limit) if n >= min_docs]


class Suggester:
    def __init__(self, phrases: List[List]):
        self.counts = {p: n for p, n in phrases}
        # (lower-cased suffix starting at a word boundary, phrase) sorted for prefix search.
        keys = []
        for phrase in self.counts:
            words = phrase.lower().split()
            for i in range(len(words)):
                keys.append((" ".join(words[i:]), phrase))
        keys.sort()
        self._keys = [k for k, _ in keys]
        self._phrases = [p for _, p in keys]

    @classmethod
    def load(cls, path: Path) -> "Suggester":
        return cls(json.loads(path.read_text()))

    def suggest(self, prefix: str, limit: int = 8) -> List[str]:
        prefix = " ".join(prefix.lower().split())
        if len(prefix) < 2:
            return []
        found = set()
        i = bisect.bisect_left(self._keys, prefix)
        while i < len(self._keys) and self._keys[i].startswith(prefix):
            found.add(self._phrases[i])
            i += 1
        return sorted(found, key=lambda p: (-self.counts[p], p))[:limit]
