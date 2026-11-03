"""Build and load the two search indexes over the chunks.

- Sparse: BM25 (bm25s, sparse matrices) on stemmed, stop-word-filtered tokens.
- Dense: bge-small-en-v1.5 sentence embeddings (384-d, L2-normalised) in a FAISS inner-product index
  (inner product of normalised vectors = cosine similarity).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List

import numpy as np

from .config import Settings

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def best_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_encoder(model_name: str, device: str):
    """Sentence-embedding model; half precision on GPU (2.5x faster on Apple MPS, same vectors)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    return model.half() if device != "cpu" else model


def tokenize_for_bm25(texts: List[str], show_progress: bool = False):
    import bm25s
    import Stemmer

    return bm25s.tokenize(texts, stopwords="en", stemmer=Stemmer.Stemmer("english"),
                          show_progress=show_progress)


def query_text_for_embedding(query: str, model_name: str) -> str:
    """bge models are trained with an instruction prefix for queries (not for passages)."""
    return BGE_QUERY_PREFIX + query if "bge" in model_name.lower() else query


def build(settings: Settings, chunk_texts: List[str], batch_size: int = 128) -> dict:
    import bm25s
    import faiss

    out = settings.index_dir
    out.mkdir(parents=True, exist_ok=True)
    timings = {}

    start = time.time()
    retriever = bm25s.BM25()
    retriever.index(tokenize_for_bm25(chunk_texts, show_progress=True), show_progress=True)
    retriever.save(str(out / "bm25"))
    timings["bm25_seconds"] = round(time.time() - start, 1)

    start = time.time()
    device = best_device()
    model = load_encoder(settings.embedding_model, device)
    embeddings = model.encode(chunk_texts, batch_size=batch_size, normalize_embeddings=True,
                              show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    timings["embedding_seconds"] = round(time.time() - start, 1)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(out / "dense.faiss"))

    info = {
        "chunks": len(chunk_texts),
        "embedding_model": settings.embedding_model,
        "embedding_dim": int(embeddings.shape[1]),
        "device": device,
        "precision": "fp16" if device != "cpu" else "fp32",
        "dense_index_mb": round(Path(out / "dense.faiss").stat().st_size / 1e6, 1),
        **timings,
    }
    (out / "index_info.json").write_text(json.dumps(info, indent=2))
    return info
