"""Settings, read from environment variables (and a local .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "gemini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash"
    max_retries: int = 2  # extra plan+execute+verify attempts after the first one fails
    llm_min_interval: float = 0.0  # seconds between LLM calls (free tiers limit requests per minute)
    trace_path: str = "logs/trace.jsonl"  # full internal reasoning of every run (never shown to users)

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()  # copies values from .env into os.environ (real env vars win)
        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", cls.llm_provider),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            gemini_model=os.getenv("GEMINI_MODEL", cls.gemini_model),
            max_retries=int(os.getenv("MAX_RETRIES", cls.max_retries)),
            llm_min_interval=float(os.getenv("LLM_MIN_INTERVAL", cls.llm_min_interval)),
            trace_path=os.getenv("TRACE_PATH", cls.trace_path),
        )
