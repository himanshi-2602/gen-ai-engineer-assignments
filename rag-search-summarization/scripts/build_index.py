"""Step 2: build the BM25 and dense (FAISS) indexes over the prepared chunks, and the auto-suggest list.

    .venv/bin/python scripts/build_index.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragsearch.config import Settings  # noqa: E402
from ragsearch.data import load_chunks  # noqa: E402
from ragsearch.index import build  # noqa: E402
from ragsearch.data import load_docs  # noqa: E402
from ragsearch.suggest import build_suggestions  # noqa: E402

settings = Settings.from_env()
chunks = load_chunks(settings)
print(f"Indexing {len(chunks['text']):,} chunks...")
print(json.dumps(build(settings, chunks["text"]), indent=2))

phrases = build_suggestions(load_docs(settings)["text"])
(settings.index_dir / "suggestions.json").write_text(json.dumps(phrases))
print(f"Auto-suggest: {len(phrases):,} phrases")
