from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar

from dotenv import load_dotenv
from google import genai
from google.genai import types

from server.utils.config import get_models_config, resolve_agent_model

load_dotenv()

_client: Optional[genai.Client] = None
_FENCE = chr(96) * 3
_T = TypeVar("_T")


def has_gemini_api_key() -> bool:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    return bool(key) and not key.startswith("your_")


def get_gemini_client() -> genai.Client:
    global _client
    if not has_gemini_api_key():
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. Please set it in .env."
        )
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if _client is None:
        _client = genai.Client(api_key=api_key)
    return _client


def _call_with_retry_and_fallback(
    primary_model: str,
    fn_for_model: Callable[[str], _T],
    fallback_key: str = "fallbacks",
) -> _T:
    fallbacks = list(get_models_config().get("models", {}).get(fallback_key, []))
    candidates = [primary_model] + [m for m in fallbacks if m != primary_model]
    last_exc: Optional[Exception] = None

    for idx, model_name in enumerate(candidates):
        try:
            return fn_for_model(model_name)
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            if any(k in msg for k in ("503", "UNAVAILABLE", "404", "NOT_FOUND")):
                print(f"[Gemini] {model_name} unavailable, trying next fallback...")
                continue
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                if idx < len(candidates) - 1:
                    print(f"[Gemini] {model_name} rate-limited, switching model...")
                    continue
                m = re.search(r"retry in (\d+(?:\.\d+)?)s", msg, re.I)
                wait_s = min(35.0, float(m.group(1)) + 1.0) if m else 15.0
                time.sleep(wait_s)
                return fn_for_model(model_name)
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError("No Gemini models available.")


def generate_text_with_config(
    agent_key: str,
    prompt: str,
    system_instruction: Optional[str] = None,
) -> str:
    client = get_gemini_client()
    model, temperature = resolve_agent_model(agent_key)
    config = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=system_instruction,
    )
    resp = _call_with_retry_and_fallback(
        model,
        lambda m: client.models.generate_content(model=m, contents=prompt, config=config),
    )
    return (resp.text or "").strip()


def generate_json_with_config(
    agent_key: str,
    prompt: str,
    system_instruction: Optional[str] = None,
) -> Dict[str, Any]:
    client = get_gemini_client()
    model, temperature = resolve_agent_model(agent_key)
    config = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json",
        system_instruction=system_instruction,
    )
    resp = _call_with_retry_and_fallback(
        model,
        lambda m: client.models.generate_content(model=m, contents=prompt, config=config),
    )
    raw = (resp.text or "{}").strip()
    cleaned = re.sub(r"^" + _FENCE + r"json\s*", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(_FENCE + r"\s*$", "", cleaned, flags=re.IGNORECASE).strip()
    return json.loads(cleaned)


def embed_text(text: str) -> List[float]:
    return embed_texts_batch([text])[0]


def embed_texts_batch(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    client = get_gemini_client()
    model = get_models_config()["models"]["embedding"]
    resp = _call_with_retry_and_fallback(
        model,
        lambda m: client.models.embed_content(model=m, contents=texts),
        fallback_key="embedding_fallbacks",
    )
    if not resp.embeddings:
        raise RuntimeError("Gemini embedding response returned no vectors.")
    return [list(e.values) for e in resp.embeddings]
