"""Data preparation: download CNN/DailyMail, clean, de-duplicate, chunk, save as Parquet."""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from .config import Settings
from .text import chunk_sentences, clean_article, clean_highlights, make_title

DATASET = "abisee/cnn_dailymail"
MIN_WORDS = 80  # articles shorter than this are stubs (photo captions, video pages)


def download(splits: Sequence[str], raw_dir: Path) -> List[Path]:
    """Download Parquet files for the given splits from the Hugging Face Hub (cached)."""
    from huggingface_hub import hf_hub_download

    return [Path(hf_hub_download(DATASET, f"3.0.0/{split}-00000-of-00001.parquet" if split != "train"
                                 else f"3.0.0/train-0000{i}-of-00003.parquet",
                                 repo_type="dataset", local_dir=str(raw_dir)))
            for split in splits for i in (range(3) if split == "train" else [0])]


def _fingerprint(text: str) -> str:
    """Hash of the lower-cased letters only: catches duplicates that differ in spacing/punctuation."""
    return hashlib.md5(re.sub(r"[^a-z]", "", text.lower()).encode()).hexdigest()


@dataclass
class PrepStats:
    raw_articles: int = 0
    exact_or_near_duplicates: int = 0
    too_short: int = 0
    documents: int = 0
    chunks: int = 0
    words_before_cleaning: int = 0
    words_after_cleaning: int = 0
    seconds: float = 0.0


def prepare(settings: Settings, splits: Sequence[str] = ("validation", "test")) -> PrepStats:
    start = time.time()
    stats = PrepStats()
    files = download(splits, settings.data_dir / "raw")

    docs: Dict[str, list] = {k: [] for k in ("doc_id", "source_id", "split", "title", "text", "highlights", "words")}
    chunks: Dict[str, list] = {k: [] for k in ("chunk_id", "doc_id", "position", "text")}
    seen = set()

    for path in files:
        split = path.name.split("-")[0]
        table = pq.read_table(path).to_pydict()
        for source_id, raw, highlights in zip(table["id"], table["article"], table["highlights"]):
            stats.raw_articles += 1
            stats.words_before_cleaning += len(raw.split())
            text = clean_article(raw)
            words = len(text.split())
            if words < MIN_WORDS:
                stats.too_short += 1
                continue
            fp = _fingerprint(text)
            if fp in seen:
                stats.exact_or_near_duplicates += 1
                continue
            seen.add(fp)

            doc_id = len(docs["doc_id"])
            for key, value in (("doc_id", doc_id), ("source_id", source_id), ("split", split),
                               ("title", make_title(text)), ("text", text),
                               ("highlights", clean_highlights(highlights)), ("words", words)):
                docs[key].append(value)
            stats.words_after_cleaning += words
            for position, chunk in enumerate(chunk_sentences(text, settings.chunk_words, settings.chunk_overlap)):
                chunks["chunk_id"].append(len(chunks["chunk_id"]))
                chunks["doc_id"].append(doc_id)
                chunks["position"].append(position)
                chunks["text"].append(chunk)

    out = settings.processed_dir
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(docs), out / "docs.parquet")
    pq.write_table(pa.table(chunks), out / "chunks.parquet")
    stats.documents, stats.chunks = len(docs["doc_id"]), len(chunks["chunk_id"])
    stats.seconds = round(time.time() - start, 1)
    (out / "prep_stats.json").write_text(json.dumps(stats.__dict__, indent=2))
    return stats


def load_docs(settings: Settings) -> Dict[str, list]:
    return pq.read_table(settings.processed_dir / "docs.parquet").to_pydict()


def load_chunks(settings: Settings) -> Dict[str, list]:
    return pq.read_table(settings.processed_dir / "chunks.parquet").to_pydict()
