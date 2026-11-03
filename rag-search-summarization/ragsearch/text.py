"""Text cleaning, sentence splitting and chunking.

Every rule here fixes a problem measured in the raw CNN/DailyMail data (see REPORT.md, section 2).
"""
from __future__ import annotations

import re
import unicodedata
from typing import List

# "(CNN)", "London (CNN) -- ", "(WIRED)", "(The Hollywood Reporter)", "(Travel + Leisure)" ... at the
# very start of an article: an optional dateline, then a short capitalised source name in brackets.
_SOURCE_TAG = re.compile(r"^\s*(?:[A-Z][\w .,'-]{0,40}?\s)?\([A-Z][\w .+&'-]{1,30}\)\s*(?:--\s*)?")
# Tokenizer artefact in the dataset: "end of sentence . Next sentence" -> "end of sentence. Next sentence".
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?])(?=\s|$)")
# Boilerplate sentences that carry no news content.
_BOILERPLATE = re.compile(
    r"(click here|read more\b|scroll down for|follow us on|follow @\w+|sign up for|"
    r"share this story|like us on facebook|all rights reserved|copyright \d{4})",
    re.IGNORECASE)
# Split on whitespace after ., ! or ? (optionally followed by a closing quote/bracket, which stays with
# the sentence) when the next sentence starts with a capital letter or digit.
_SENTENCE_END = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"')\]]))\s+(?=[\"'(\[]?[A-Z0-9])")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Unicode NFKC (curly quotes, odd spaces), collapse whitespace."""
    return _WS.sub(" ", unicodedata.normalize("NFKC", text)).strip()


# A full stop after these is not a sentence end ("Dr. Smith", "the U.S. Army", "10 a.m. Monday").
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "st", "sr", "jr", "gen", "gov", "sen", "rep", "lt", "col", "capt",
    "sgt", "cpl", "maj", "rev", "hon", "pres", "no", "vs", "inc", "co", "corp", "ltd", "jan", "feb",
    "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "u.s", "u.k", "u.n", "e.g",
    "i.e", "a.m", "p.m", "mt", "ft", "ave", "est", "approx", "dept", "univ", "d.c", "l.a",
}


def _ends_with_abbreviation(piece: str) -> bool:
    last = piece.rsplit(None, 1)[-1].rstrip(".").lstrip("(\"'").lower()
    return last in _ABBREVIATIONS or (len(last) == 1 and last.isalpha())  # initials: "J. K. Rowling"


def split_sentences(text: str) -> List[str]:
    """Light-weight rule-based sentence splitter (good enough for news prose)."""
    sentences: List[str] = []
    for piece in _SENTENCE_END.split(text):
        piece = piece.strip()
        if not piece:
            continue
        if sentences and _ends_with_abbreviation(sentences[-1]):
            sentences[-1] += " " + piece  # the previous "sentence" ended with an abbreviation: rejoin
        else:
            sentences.append(piece)
    return sentences


def clean_article(raw: str) -> str:
    text = normalize(raw)
    text = _SOURCE_TAG.sub("", text, count=1)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    sentences = [s for s in split_sentences(text) if not _BOILERPLATE.search(s)]
    return " ".join(sentences)


def clean_highlights(raw: str) -> str:
    """Reference summary: highlight lines joined into sentences."""
    lines = [_SPACE_BEFORE_PUNCT.sub(r"\1", normalize(line)) for line in raw.split("\n") if line.strip()]
    return " ".join(line if line.endswith((".", "!", "?")) else line + "." for line in lines)


def make_title(text: str, max_words: int = 18) -> str:
    """Pseudo-title for display: the first sentence, shortened (the dataset has no titles)."""
    first = split_sentences(text)[0] if text else ""
    words = first.split()
    return " ".join(words[:max_words]) + ("…" if len(words) > max_words else "")


def chunk_sentences(text: str, chunk_words: int = 200, overlap_words: int = 50) -> List[str]:
    """Split into passages of about `chunk_words` words, made of whole sentences.

    Consecutive chunks share about `overlap_words` words of trailing sentences, so a fact that
    straddles a boundary is fully contained in at least one chunk.
    """
    sentences = split_sentences(text)
    chunks: List[str] = []
    current: List[str] = []
    count = 0
    for sentence in sentences:
        n = len(sentence.split())
        if current and count + n > chunk_words:
            chunks.append(" ".join(current))
            # Carry the last sentences (up to overlap_words) into the next chunk.
            carry, carried = [], 0
            for prev in reversed(current):
                w = len(prev.split())
                if carried + w > overlap_words:
                    break
                carry.insert(0, prev)
                carried += w
            current, count = carry, carried
        current.append(sentence)
        count += n
    if current and (not chunks or count > overlap_words):
        chunks.append(" ".join(current))
    return chunks
