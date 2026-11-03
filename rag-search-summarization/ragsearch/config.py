"""Paths and settings (overridable with environment variables / a local .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    data_dir: Path = ROOT / "data"
    # Embedding model (bi-encoder) for dense retrieval, and cross-encoder for re-ranking.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Chunking: passages of about `chunk_words` words, overlapping by about `chunk_overlap` words.
    chunk_words: int = 200
    chunk_overlap: int = 50
    # LLM (summaries and evaluation queries). Comma-separated fallback list.
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash,gemini-3-flash-preview,gemini-3.1-flash-lite,gemini-3.5-flash-lite"
    llm_min_interval: float = 0.0

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(ROOT / ".env")
        return cls(
            data_dir=Path(os.getenv("DATA_DIR", cls.data_dir)),
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            reranker_model=os.getenv("RERANKER_MODEL", cls.reranker_model),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            gemini_model=os.getenv("GEMINI_MODEL", cls.gemini_model),
            llm_min_interval=float(os.getenv("LLM_MIN_INTERVAL", cls.llm_min_interval)),
        )
