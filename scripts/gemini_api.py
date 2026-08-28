# gemini_api.py
import time
import random
from typing import Optional

from google import genai


# ================== CONFIG ==================
API_KEY = "API_KEY"   # replce with actual api key
MODEL_NAME = "gemini-3-flash-preview"

MAX_RETRIES = 2            # keeping low for free tier
BACKOFF_BASE = 1.0         # seconds
SOFT_TIMEOUT = 25          # seconds (entire call)
# ============================================


# Reuse client across calls (important)
_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=API_KEY)
    return _client


def _classify_error(e: Exception) -> str:
    msg = str(e).upper()

    if any(k in msg for k in ["429", "RESOURCE_EXHAUSTED", "QUOTA", "RATE"]):
        return "quota"

    if any(k in msg for k in ["503", "UNAVAILABLE", "500", "INTERNAL", "TIMEOUT", "DEADLINE"]):
        return "transient"

    return "other"


def _backoff(attempt: int):
    # exponential backoff + jitter
    sleep_s = (BACKOFF_BASE * (2 ** attempt)) + random.uniform(0, 0.3)
    time.sleep(sleep_s)


def api_call(message: str) -> str:
    if not message or not isinstance(message, str):
        raise ValueError("api_call requires a non-empty prompt string.")

    client = _get_client()
    start = time.time()
    last_err = None

    for attempt in range(MAX_RETRIES + 1):
        if (time.time() - start) > SOFT_TIMEOUT:
            raise RuntimeError("TIMEOUT: Gemini request took too long.")

        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=message,
            )

            text = getattr(response, "text", None)
            if not text or not str(text).strip():
                raise RuntimeError("EMPTY_RESPONSE from Gemini.")

            return str(text)

        except Exception as e:
            last_err = e
            kind = _classify_error(e)

            if kind == "quota":
                # No retries for quota errors
                raise RuntimeError(f"QUOTA: {e}") from e

            if kind == "transient" and attempt < MAX_RETRIES:
                _backoff(attempt)
                continue

            # Other errors or out of retries
            raise

    # Should never reach here
    raise last_err if last_err else RuntimeError("Unknown Gemini error")
