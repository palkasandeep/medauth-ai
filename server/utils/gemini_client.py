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


def _call_with_retry(fn: Callable[[], _T]) -> _T:
    for attempt in range(3):
        try:
            return fn()
        except Exception as exc:
            msg = str(exc)
            if ("429" in msg or "RESOURCE_EXHAUSTED" in msg) and attempt < 2:
                match = re.search(r"retry in (\d+(?:\.\d+)?)s", msg, re.I)
                wait_s = min(45.0, float(match.group(1)) + 2.0) if match else 20.0
                time.sleep(wait_s)
                continue
            raise
    return fn()


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
    resp = _call_with_retry(
        lambda: client.models.generate_content(model=model, contents=prompt, config=config)
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
    resp = _call_with_retry(
        lambda: client.models.generate_content(model=model, contents=prompt, config=config)
    )
    raw = (resp.text or "{}").strip()
    cleaned = re.sub(r"^" + _FENCE + r"json\s*", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(_FENCE + r"\s*$", "", cleaned, flags=re.IGNORECASE).strip()
    return json.loads(cleaned)


def embed_text(text: str) -> List[float]:
    client = get_gemini_client()
    model = get_models_config()["models"]["embedding"]
    resp = _call_with_retry(
        lambda: client.models.embed_content(model=model, contents=text)
    )
    if not resp.embeddings or not resp.embeddings[0].values:
        raise RuntimeError("Gemini embedding response returned no vector values.")
    return list(resp.embeddings[0].values)
