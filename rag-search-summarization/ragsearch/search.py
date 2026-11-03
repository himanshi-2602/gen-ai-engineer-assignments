"""Search engine: BM25, dense embeddings, hybrid fusion and cross-encoder re-ranking.

Retrieval works on chunks (passages) and results are aggregated to documents: a document's rank in
a method is the rank of its best chunk, and that chunk is shown as the excerpt.

Methods
- "bm25":   keyword matching (exact terms, names, numbers).
- "dense":  semantic similarity of bge-small embeddings (paraphrases, synonyms).
- "hybrid": Reciprocal Rank Fusion of both:  score(d) = sum over methods of 1 / (K + rank_m(d)).
            RRF uses ranks, not raw scores, so BM25 and cosine scores need no calibration.
- "hybrid_rerank": hybrid, then the top candidates are re-scored by a cross-encoder that reads
            the query and passage together (more accurate, slower, so only on a short list).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import Settings
from .data import load_chunks, load_docs
from .index import best_device, load_encoder, query_text_for_embedding

METHODS = ("bm25", "dense", "hybrid", "hybrid_rerank")
RRF_K = 60            # standard constant from Cormack et al. (2009)
CANDIDATE_CHUNKS = 200  # chunks retrieved per method before aggregating to documents
RERANK_TOP = 30       # documents re-scored by the cross-encoder
MAX_RESULTS = 100     # documents available for pagination


@dataclass
class DocResult:
    doc_id: int
    rank: int
    score: float
    title: str
    excerpt: str
    chunk_id: int
    words: int
    matched_by: List[str] = field(default_factory=list)  # which retrievers found it ("bm25", "dense")


@dataclass
class SearchResponse:
    query: str
    method: str
    total: int
    page: int
    page_size: int
    results: List[DocResult]
    timings_ms: Dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


class SearchEngine:
    def __init__(self, settings: Settings, load_reranker: bool = True):
        import bm25s
        import faiss
        from sentence_transformers import CrossEncoder

        start = time.time()
        self.settings = settings
        self.docs = load_docs(settings)
        chunks = load_chunks(settings)
        self.chunk_text: List[str] = chunks["text"]
        self.chunk_doc = np.asarray(chunks["doc_id"], dtype=np.int64)
        self.bm25 = bm25s.BM25.load(str(settings.index_dir / "bm25"))
        self.dense = faiss.read_index(str(settings.index_dir / "dense.faiss"))
        device = best_device()
        self.encoder = load_encoder(settings.embedding_model, device)
        self.reranker = CrossEncoder(settings.reranker_model, device=device) if load_reranker else None
        self.load_seconds = round(time.time() - start, 1)

    # ---------- chunk-level retrievers: return [(chunk_index, score)] best first ----------

    def bm25_chunks(self, query: str, k: int = CANDIDATE_CHUNKS) -> List[Tuple[int, float]]:
        import bm25s
        import Stemmer

        tokens = bm25s.tokenize([query], stopwords="en", stemmer=Stemmer.Stemmer("english"),
                                return_ids=False, show_progress=False)
        if not tokens or not tokens[0]:
            return []  # the query was only stop words
        ids, scores = self.bm25.retrieve(tokens, k=min(k, len(self.chunk_text)), show_progress=False)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if s > 0]

    def dense_chunks(self, query: str, k: int = CANDIDATE_CHUNKS) -> List[Tuple[int, float]]:
        vector = self.encoder.encode([query_text_for_embedding(query, self.settings.embedding_model)],
                                     normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        scores, ids = self.dense.search(vector, k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    # ---------- document-level ranking ----------

    def _to_docs(self, chunk_hits: List[Tuple[int, float]]) -> List[Tuple[int, float, int]]:
        """Keep each document's best chunk: [(doc_id, score, chunk_index)] in rank order."""
        seen, out = set(), []
        for chunk, score in chunk_hits:
            doc = int(self.chunk_doc[chunk])
            if doc not in seen:
                seen.add(doc)
                out.append((doc, score, chunk))
        return out

    def rank(self, query: str, method: str = "hybrid") -> Tuple[List[Tuple[int, float, int, List[str]]], Dict]:
        """Full ranking for a query: [(doc_id, score, best_chunk, matched_by)] and timings in ms."""
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        timings: Dict[str, float] = {}
        lists: Dict[str, List[Tuple[int, float, int]]] = {}

        if method in ("bm25", "hybrid", "hybrid_rerank"):
            t = time.perf_counter()
            lists["bm25"] = self._to_docs(self.bm25_chunks(query))
            timings["bm25"] = (time.perf_counter() - t) * 1000
        if method in ("dense", "hybrid", "hybrid_rerank"):
            t = time.perf_counter()
            lists["dense"] = self._to_docs(self.dense_chunks(query))
            timings["dense"] = (time.perf_counter() - t) * 1000

        if method in ("bm25", "dense"):
            ranking = [(d, s, c, [method]) for d, s, c in lists[method]]
        else:
            ranking = rrf_fuse(lists)
            if method == "hybrid_rerank" and self.reranker is not None and ranking:
                t = time.perf_counter()
                ranking = self._rerank(query, ranking)
                timings["rerank"] = (time.perf_counter() - t) * 1000
        return ranking[:MAX_RESULTS], {k: round(v, 1) for k, v in timings.items()}

    def _rerank(self, query: str, ranking):
        head, tail = ranking[:RERANK_TOP], ranking[RERANK_TOP:]
        scores = self.reranker.predict([(query, self.chunk_text[c]) for _, _, c, _ in head], show_progress_bar=False)
        order = np.argsort(-np.asarray(scores))
        reranked = [(head[i][0], float(scores[i]), head[i][2], head[i][3]) for i in order]
        return reranked + tail

    def search(self, query: str, method: str = "hybrid", page: int = 1, page_size: int = 10) -> SearchResponse:
        start = time.perf_counter()
        ranking, timings = self.rank(query, method)
        page = max(1, page)
        window = ranking[(page - 1) * page_size: page * page_size]
        results = [
            DocResult(doc_id=d, rank=(page - 1) * page_size + i + 1, score=round(s, 4),
                      title=self.docs["title"][d], excerpt=self.chunk_text[c], chunk_id=c,
                      words=self.docs["words"][d], matched_by=m)
            for i, (d, s, c, m) in enumerate(window)
        ]
        timings["total"] = round((time.perf_counter() - start) * 1000, 1)
        return SearchResponse(query, method, len(ranking), page, page_size, results, timings)

    def document(self, doc_id: int) -> dict:
        return {key: self.docs[key][doc_id] for key in ("doc_id", "title", "text", "words", "split", "source_id")}


def rrf_fuse(lists: Dict[str, List[Tuple[int, float, int]]], k: int = RRF_K):
    """Reciprocal Rank Fusion of document rankings from several retrievers."""
    scores: Dict[int, float] = {}
    best_chunk: Dict[int, int] = {}
    matched: Dict[int, List[str]] = {}
    for name, ranking in lists.items():
        for rank, (doc, _, chunk) in enumerate(ranking, 1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
            matched.setdefault(doc, []).append(name)
            best_chunk.setdefault(doc, chunk)  # chunk from the first retriever that found it
    order = sorted(scores, key=lambda d: -scores[d])
    return [(d, scores[d], best_chunk[d], matched[d]) for d in order]
