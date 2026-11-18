"""Closed API side row (DEVIATIONS D4): gpt-5-nano through the OpenAI Chat Completions API, same interface as
mlxcall.complete. NEVER a status agent: it is a pipeline check of the frozen rule on real agent transcripts.

One agent step = ONE call: a system message = the system prompt, one user message = the JSON prompt (task, tools,
reply format, full transcript), exactly as the MLX backend. gpt-5 models take no temperature or top_p (the API's
default sampling, temperature 1); reasoning_effort 'minimal'. The run id is sent as prompt_cache_key, so the growing
transcript of one run hits OpenAI's automatic prefix cache.

Tokens: usage.prompt_tokens (of which prompt_tokens_details.cached_tokens), usage.completion_tokens (includes
completion_tokens_details.reasoning_tokens). Dollars per call from PRICE (USD per 1M tokens, OpenAI price page for
gpt-5-nano: input 0.05, cached input 0.005, output 0.40). Cache: one JSON file per call under <cache>/<model>/<sha256>.json,
keyed by model, run id, seed, system and prompt. SPEND_CAP_USD: run.py stops starting runs once spent() reaches it.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

API = "https://api.openai.com/v1/chat/completions"
GEN_SETTINGS_VERSION = "openai-v1"
MAX_COMPLETION_TOKENS = 4096
CACHE = Path(os.environ.get("MAZE_CACHE", str(Path(__file__).resolve().parent.parent / "data" / "cache")))
SPEND_CAP_USD = float(os.environ.get("MAZE_SPEND_CAP_USD", "2.00"))

MODELS: dict[str, dict[str, Any]] = {
    "gpt-5-nano": {"api_id": "gpt-5-nano", "reasoning_effort": "minimal",
                   "price": {"input": 0.05, "cached": 0.005, "output": 0.40}},
}

_LOCK = threading.Lock()
_SPENT: dict[str, float] = {}


def gen_settings(name: str) -> dict[str, Any]:
    s = MODELS[name]
    return {"backend": "openai-chat", "version": GEN_SETTINGS_VERSION, "api_id": s["api_id"],
            "reasoning_effort": s["reasoning_effort"], "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "price_per_m": s["price"], "spend_cap_usd": SPEND_CAP_USD, "sampling": "API default (temperature 1)"}


def cost(name: str, usage: dict) -> float:
    p = MODELS[name]["price"]
    pt = usage.get("prompt_tokens", 0)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    out = usage.get("completion_tokens", 0)
    return ((pt - cached) * p["input"] + cached * p["cached"] + out * p["output"]) / 1e6


def spent(name: str) -> float:
    """USD billed so far for this model: every cached response on disk (each was one billed call)."""
    with _LOCK:
        if name not in _SPENT:
            tot = 0.0
            for f in (CACHE / name).glob("*.json"):
                try:
                    tot += cost(name, json.loads(f.read_text()).get("usage", {}))
                except Exception:
                    pass
            _SPENT[name] = tot
        return _SPENT[name]


def _post(body: dict) -> dict:
    key = os.environ["OPENAI_API_KEY"]
    data = json.dumps(body).encode()
    delay = 5.0
    err = ""
    for attempt in range(10):
        req = urllib.request.Request(API, data=data, headers={"Content-Type": "application/json",
                                                              "Authorization": f"Bearer {key}"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return {**json.load(r), "_attempts": attempt + 1}
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}: {e.read()[:300]!r}"
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(delay)
                delay = min(delay * 1.6, 60)
                continue
            raise RuntimeError(err) from e
        except (urllib.error.URLError, TimeoutError) as e:
            err = repr(e)
            time.sleep(delay)
            delay = min(delay * 1.6, 60)
    raise RuntimeError(f"openai: retries exhausted ({err})")


def complete(name: str, system: str, prompt: str, seed: int, run_id: str = "") -> tuple[str, str | None, dict[str, Any]]:
    """One agent step. Returns (reasoning, final_text or None, meta) like mlxcall.complete."""
    spec = MODELS[name]
    body = {"model": spec["api_id"], "reasoning_effort": spec["reasoning_effort"],
            "max_completion_tokens": MAX_COMPLETION_TOKENS, "prompt_cache_key": run_id or None,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
    key = hashlib.sha256(json.dumps([name, run_id, seed, system, prompt]).encode()).hexdigest()
    cpath = CACHE / name / f"{key}.json"
    cached = cpath.exists()
    t0 = time.time()
    if cached:
        resp = json.loads(cpath.read_text())
    else:
        resp = _post(body)
        cpath.parent.mkdir(parents=True, exist_ok=True)
        cpath.write_text(json.dumps(resp))
        spent(name)
        with _LOCK:
            _SPENT[name] += cost(name, resp.get("usage", {}))
    dt = time.time() - t0
    ch = (resp.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    finish = ch.get("finish_reason")
    final = msg.get("content")
    final = final.strip() if isinstance(final, str) and final.strip() else None
    u = resp.get("usage", {})
    rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
    meta = {"seed": seed, "prompt_tokens": u.get("prompt_tokens", 0), "new_tokens": u.get("completion_tokens", 0),
            "reasoning_tokens": rt, "cached_prompt_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            "fed_tokens": u.get("prompt_tokens", 0), "stopped": finish == "stop", "finish_reason": finish,
            "harmony_call": False, "cap_hit": finish == "length", "forced_close": False, "latency_s": round(dt, 3),
            "cached": cached, "attempts": resp.get("_attempts"), "model_version": resp.get("model"),
            "response_id": resp.get("id"), "cost_usd": round(cost(name, u), 7), "usage": u,
            "prompt_sha256": hashlib.sha256((system + "\n" + prompt).encode()).hexdigest()}
    return "", final, {**meta, "raw": json.dumps(msg)}
