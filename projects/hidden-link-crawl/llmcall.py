"""Cached multi-provider model calls with per-call cost logging.

Copied from projects/scoped-recording/agent.py (branch swarm/defense-1107-scoped-recording), local MLX path removed.
Temperature is the provider default for every API model; each run has its own URL token, so no two runs share a prompt.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

MAX_TOKENS = 400  # visible-answer cap for non-reasoning models
MAX_TOKENS_REASONING = 1500  # gpt-5*: reasoning tokens count inside the cap (Kasper #9021)
CACHE_DIR = Path(__file__).resolve().parent / "raw" / "cache"
COST_PATH = Path(__file__).resolve().parent / "raw" / "cost.jsonl"

# Generation settings, recorded per call (Kasper #9021): temperature is the provider default for API models.
GEN_SETTINGS_VERSION = "v2"

# USD per million tokens (input, output), list prices; models without a confirmed list price use a deliberately high
# estimate so the logged spend over-counts rather than under-counts.
PRICE_PER_MILLION = {
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5.4-mini": (0.75, 4.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.5-flash-lite": (0.25, 1.50),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-haiku-5-5": (1.00, 5.00),
}


def _provider(model: str) -> str:
    m = model.lower()
    if m.startswith("gpt-") or m.startswith("o"):
        return "openai"
    if m.startswith("gemini-"):
        return "gemini"
    if m.startswith("claude-"):
        return "anthropic"
    raise ValueError(f"unknown model provider for {model}")


def _is_reasoning_openai(model: str) -> bool:
    return model.startswith("gpt-5")


def gen_settings(model: str) -> dict[str, Any]:
    p = _provider(model)
    if p == "openai" and _is_reasoning_openai(model):
        return {"max_completion_tokens": MAX_TOKENS_REASONING, "reasoning_effort": "minimal" if model.startswith("gpt-5-") else "low", "temperature": "provider default"}
    if p == "openai":
        return {"max_completion_tokens": MAX_TOKENS, "temperature": "provider default"}
    if p == "gemini":
        return {"max_output_tokens": MAX_TOKENS_REASONING, "thinking_budget": 0, "temperature": "provider default"}
    if p == "anthropic":
        return {"max_tokens": MAX_TOKENS_REASONING, "temperature": "provider default"}
    raise ValueError(model)


def _cache_key(model: str, system: str, prompt: str, max_tokens: int) -> str:
    raw = f"{model}\0{system}\0{prompt}\0{max_tokens}".encode()
    return hashlib.sha256(raw).hexdigest()


def _estimate_cost(model: str, inp: int, out: int) -> float:
    pi, po = PRICE_PER_MILLION.get(model, (0.50, 2.00))
    return inp * pi / 1_000_000 + out * po / 1_000_000


_COST_LOCK = threading.Lock()


def _append_cost(rec: dict[str, Any]) -> None:
    COST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _COST_LOCK, COST_PATH.open("a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")


def complete_cached(model: str, prompt: str, system: str) -> tuple[str, dict[str, Any]]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    settings = gen_settings(model)
    key = _cache_key(model, system, prompt, json.dumps(settings, sort_keys=True))
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        data = json.loads(path.read_text())
        data["cache_hit"] = True
        return data["text"], data
    provider = _provider(model)
    t0 = time.time()
    text = ""
    usage: dict[str, Any] = {"input_tokens": 0, "output_tokens": 0}
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(max_retries=6)
        kw: dict[str, Any] = {"max_completion_tokens": settings["max_completion_tokens"]}
        if "reasoning_effort" in settings:
            kw["reasoning_effort"] = settings["reasoning_effort"]
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            **kw,
        )
        msg = resp.choices[0].message
        text = (msg.content or getattr(msg, "refusal", None) or "")
        if getattr(resp, "usage", None):
            usage = {"input_tokens": resp.usage.prompt_tokens or 0, "output_tokens": resp.usage.completion_tokens or 0}
    elif provider == "anthropic":
        from anthropic import Anthropic
        client = Anthropic(max_retries=6)
        resp = client.messages.create(
            model=model,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=settings["max_tokens"],
        )
        parts = [getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text"]
        text = "\n".join(parts)
        if getattr(resp, "usage", None):
            usage = {"input_tokens": resp.usage.input_tokens or 0, "output_tokens": resp.usage.output_tokens or 0}
    elif provider == "gemini":
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))

        def cfg(think: int | None) -> Any:
            kw = {"max_output_tokens": settings["max_output_tokens"], "system_instruction": system}
            if think is not None:
                kw["thinking_config"] = types.ThinkingConfig(thinking_budget=think)
            return types.GenerateContentConfig(**kw)

        resp = None
        for attempt in range(6):
            try:
                try:
                    resp = client.models.generate_content(model=model, contents=prompt, config=cfg(0))
                except Exception as e:
                    if not any(w in str(e).lower() for w in ("thinking", "budget", "invalid_argument")):
                        raise
                    usage["thinking_budget_rejected"] = True
                    resp = client.models.generate_content(model=model, contents=prompt, config=cfg(None))
                break
            except Exception as e:
                if attempt == 5 or not any(c in str(e) for c in ("429", "500", "503", "UNAVAILABLE", "RESOURCE_EXHAUSTED")):
                    raise
                time.sleep(2 ** attempt)
        try:
            text = resp.text or ""
        except Exception:
            text = ""
        meta = getattr(resp, "usage_metadata", None)
        if meta:
            out = (getattr(meta, "candidates_token_count", 0) or 0) + (getattr(meta, "thoughts_token_count", 0) or 0)
            usage.update({"input_tokens": getattr(meta, "prompt_token_count", 0) or 0, "output_tokens": out})
    cost = _estimate_cost(model, int(usage["input_tokens"]), int(usage["output_tokens"]))
    data = {
        "key": key,
        "model": model,
        "provider": provider,
        "settings": settings,
        "text": text,
        "usage": usage,
        "cost_usd_est": cost,
        "cache_hit": False,
        "latency_s": round(time.time() - t0, 3),
    }
    tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(data, sort_keys=True))
    tmp.replace(path)
    _append_cost({k: data[k] for k in ["key", "model", "provider", "usage", "cost_usd_est", "latency_s"]})
    return text, data
