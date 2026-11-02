"""LLM clients.

The rest of the agent only depends on the `LLMClient` interface (one method, `generate`),
so the real Gemini client, a fake one for tests, or another provider can be swapped in
without touching the planner, executor or verifier.
"""
from __future__ import annotations

import time
from typing import Dict, Iterable, List, Protocol

from .config import Settings


class LLMError(RuntimeError):
    """Any failure talking to an LLM (network, quota, empty response...)."""


class LLMClient(Protocol):
    """Anything with this method can be used as the agent's LLM."""

    def generate(self, system: str, user: str, *, json_mode: bool = False, temperature: float = 0.0) -> str:
        """Send a system prompt + user message, return the model's text reply.

        json_mode=True asks the model to reply with valid JSON only.
        """
        ...


class GeminiClient:
    """Google Gemini via the official `google-genai` SDK, with retries and model fallback.

    `model` may be a comma-separated list, e.g. "gemini-3.5-flash,gemini-3-flash-preview".
    Temporary errors (per-minute rate limit, overload, timeout) are retried with backoff on the
    same model. If a model's daily quota is used up, or it keeps failing, the client switches to
    the next model in the list for the rest of the process.
    """

    RETRYABLE_CODES = {429, 500, 503, 504}  # rate limited, server error, overloaded, timed out

    def __init__(self, api_key: str, model: str, max_attempts: int = 3, min_interval: float = 0.0,
                 timeout_seconds: int = 60):
        if not api_key or api_key == "paste-your-key-here":
            raise LLMError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
        from google import genai  # imported here so tests with MockLLM don't need the SDK
        from google.genai import types

        # Without a timeout a stuck request can hang forever; 504s are then retried below.
        self._client = genai.Client(api_key=api_key,
                                    http_options=types.HttpOptions(timeout=timeout_seconds * 1000))
        self.models = [m.strip() for m in model.split(",") if m.strip()]
        self._current = 0
        self.max_attempts = max_attempts  # per model
        self.min_interval = min_interval  # seconds between calls, to stay under free-tier rate limits
        self._last_call = 0.0

    @property
    def model(self) -> str:
        """The model currently in use."""
        return self.models[self._current]

    def generate(self, system: str, user: str, *, json_mode: bool = False, temperature: float = 0.0) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            response_mime_type="application/json" if json_mode else "text/plain",
        )
        errors_seen = []
        while True:
            try:
                return self._generate_with_retries(user, config)
            except _SwitchModel as reason:
                errors_seen.append(f"{self.model}: {reason}")
                if self._current + 1 >= len(self.models):
                    raise LLMError("All Gemini models failed. " + " | ".join(errors_seen)) from None
                self._current += 1  # fall back to the next model and keep using it

    def _generate_with_retries(self, user: str, config) -> str:
        from google.genai import errors

        for attempt in range(1, self.max_attempts + 1):
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            try:
                response = self._client.models.generate_content(model=self.model, contents=user, config=config)
            except errors.APIError as err:
                message = str(err.message or "")
                if err.code == 429 and "per day" in message.lower():
                    raise _SwitchModel("daily free-tier quota used up")
                if err.code in self.RETRYABLE_CODES:
                    if attempt < self.max_attempts:
                        # Per-minute limit or overload: wait 5s, 10s, ... and retry the same model.
                        time.sleep(min(60, 5 * 2 ** (attempt - 1)))
                        continue
                    raise _SwitchModel(f"error {err.code} after {attempt} attempts")
                if err.code == 402:
                    raise LLMError("Gemini API error 402: the API key's project has no credits left "
                                   "(billing is enabled, so the free tier does not apply). Add credits or "
                                   "use a key from a project without billing.") from err
                raise LLMError(f"Gemini API error {err.code}: {message[:300]}") from err
            if not response.text:
                raise LLMError("Gemini returned an empty response.")
            return response.text
        raise _SwitchModel("no attempts left")


class _SwitchModel(Exception):
    """Internal signal: give up on the current model and try the next one."""


class MockLLM:
    """Fake LLM for tests: returns pre-written responses in order and records every call."""

    def __init__(self, responses: Iterable[str]):
        self.responses: List[str] = list(responses)
        self.calls: List[Dict] = []

    def generate(self, system: str, user: str, *, json_mode: bool = False, temperature: float = 0.0) -> str:
        self.calls.append({"system": system, "user": user, "json_mode": json_mode})
        if not self.responses:
            raise LLMError("MockLLM ran out of responses")
        return self.responses.pop(0)


def make_llm(settings: Settings) -> LLMClient:
    """Factory: build the LLM client named in the settings."""
    if settings.llm_provider == "gemini":
        return GeminiClient(settings.gemini_api_key, settings.gemini_model,
                            min_interval=settings.llm_min_interval)
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider!r}")
