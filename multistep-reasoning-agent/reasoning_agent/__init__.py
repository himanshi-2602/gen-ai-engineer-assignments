"""Multi-step reasoning agent: planner -> executor -> verifier, with retries."""
import warnings

# Python 3.9 is past end of life; the Google SDK warns about it on import. Harmless here.
warnings.filterwarnings("ignore", message=".*Python version 3.9 past its end of life.*")
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")

from .agent import solve  # noqa: E402,F401  (from reasoning_agent import solve)
