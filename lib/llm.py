"""Multi-provider LLM API with an on-disk cache so repeated research runs are cheap.

    from lib import llm
    text = llm.complete("Summarize this dataset...", system="You are terse.")

Routes by model id: claude-* -> Anthropic, gpt-*/o1*/o3*/o4* -> OpenAI,
gemini-* -> Google. Set the matching key in the environment (ANTHROPIC_API_KEY,
OPENAI_API_KEY, GEMINI_API_KEY). Cache lives in .llm_cache/ (gitignored, keyed by
model+system+prompt+max_tokens); pass cache=False to force a fresh call.
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


def _provider(model: str) -> str:
    m = model.lower()
    if m.startswith(("gpt", "o1", "o3", "o4", "chatgpt")):
        return "openai"
    if m.startswith("gemini"):
        return "gemini"
    return "anthropic"


def _anthropic(model, system, prompt, max_tokens) -> str:
    from anthropic import Anthropic
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    kwargs: dict = {"model": model, "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}]}
    if system:
        kwargs["system"] = system
    resp = client.messages.create(**kwargs)
    texts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    return texts[0] if texts else ""


def _openai(model, system, prompt, max_tokens) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]

    def call(**kw):
        return client.chat.completions.create(model=model, messages=msgs, **kw)

    try:
        r = call(max_completion_tokens=max_tokens)
    except Exception as e:  # older chat models want max_tokens instead
        s = str(e)
        if "max_completion_tokens" in s or "max_tokens" in s or "Unsupported" in s:
            r = call(max_tokens=max_tokens)
        else:
            raise
    msg = r.choices[0].message
    return (msg.content or getattr(msg, "refusal", None) or "")


def _gemini(model, system, prompt, max_tokens) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"])
    # Turn OFF the API-level safety filter so we measure the MODEL's own refusal,
    # not Google's content filter (matches how Claude/GPT are queried here).
    cats = ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
    safety = [types.SafetySetting(category=c, threshold="BLOCK_NONE") for c in cats]
    mot = max(max_tokens, 2048)  # leave headroom for 2.5 "thinking"

    def cfg(think):
        kw: dict = {"max_output_tokens": mot, "safety_settings": safety}
        if system:
            kw["system_instruction"] = system
        if think is not None:
            kw["thinking_config"] = types.ThinkingConfig(thinking_budget=think)
        return types.GenerateContentConfig(**kw)

    try:                       # try with thinking disabled (flash-class)
        r = client.models.generate_content(model=model, contents=prompt, config=cfg(0))
    except Exception:          # pro-class may reject thinking_budget=0
        r = client.models.generate_content(model=model, contents=prompt, config=cfg(None))
    try:
        return (r.text or "")
    except Exception:
        return ""


_PROVIDERS = {"anthropic": _anthropic, "openai": _openai, "gemini": _gemini}


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

    text = _PROVIDERS[_provider(model)](model, system, prompt, max_tokens)

    if cache:
        _CACHE.mkdir(exist_ok=True)
        cf.write_text(json.dumps({"model": model, "text": text}))
    return text
