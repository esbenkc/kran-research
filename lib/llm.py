"""Anthropic API with an on-disk cache so repeated research runs are cheap and deterministic.

    from lib import llm
    text = llm.complete("Summarize this dataset...", system="You are terse.")

Set ANTHROPIC_API_KEY in the environment. Cache lives in .llm_cache/ (gitignored); pass
cache=False to force a fresh call.
"""
from __future__ import annotations

import os
import json
import hashlib
from pathlib import Path

DEFAULT_MODEL = "claude-sonnet-4-6"  # cheap default for batch research; override per call
_CACHE = Path(__file__).resolve().parent.parent / ".llm_cache"


def _key(model: str, system, prompt: str, max_tokens: int) -> str:
    raw = f"{model}\x00{system}\x00{prompt}\x00{max_tokens}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]


def complete(
    prompt: str,
    system: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2048,
    cache: bool = True,
) -> str:
    """Return the model's text for a single-turn prompt, cached by (model, system, prompt, max_tokens)."""
    cf = _CACHE / f"{_key(model, system, prompt, max_tokens)}.json"
    if cache and cf.exists():
        return json.loads(cf.read_text())["text"]

    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    text = client.messages.create(**kwargs).content[0].text

    if cache:
        _CACHE.mkdir(exist_ok=True)
        cf.write_text(json.dumps({"model": model, "text": text}))
    return text
