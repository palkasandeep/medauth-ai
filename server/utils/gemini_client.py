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
) -> _T:
    fallbacks = list(get_models_config().get("models", {}).get("fallbacks", []))
    candidates = [primary_model] + [m for m in fallbacks if m != primary_model]
    last_exc: Optional[Exception] = None

    for idx, model_name in enumerate(candidates):
        for attempt in range(2):
            try:
                return fn_for_model(model_name)
            except Exception as exc:
                last_exc = exc
                msg = str(exc)
                if "503" in msg or "UNAVAILABLE" in msg:
                    if attempt == 0:
                        time.sleep(2.0)
                        continue
                    break  # try next fallback model
                if "404" in msg or "NOT_FOUND" in msg:
                    break  # try next fallback model
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    if attempt == 0 and idx == len(candidates) - 1:
                        m = re.search(r"retry in (\d+(?:\.\d+)?)s", msg, re.I)
                        time.sleep(min(45.0, float(m.group(1)) + 2.0) if m else 25.0)
                        continue
                    break  # try next fallback model first (each model has its own RPM quota!)
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
    client = get_gemini_client()
    model = get_models_config()["models"]["embedding"]
    resp = _call_with_retry_and_fallback(
        model,
        lambda m: client.models.embed_content(model=m, contents=text),
    )
    if not resp.embeddings or not resp.embeddings[0].values:
        raise RuntimeError("Gemini embedding response returned no vector values.")
    return list(resp.embeddings[0].values)
