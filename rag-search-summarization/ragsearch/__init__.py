"""Document search (hybrid BM25 + embeddings) and LLM summarization over a news corpus."""
import warnings

# Python 3.9 is past end of life; Google / urllib3 warn about it on import. Harmless here.
warnings.filterwarnings("ignore", message=".*Python version 3.9 past its end of life.*")
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")
