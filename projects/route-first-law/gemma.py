"""Open-weight Gemma 4 agents through the Gemini API (generateContent). Shared by both harnesses (web/ and shell/).

Copied from maze-cost (room defense-1118-maze-cost, projects/maze-cost/web/geminicall.py). Changes for route-first-law:
the cache lives in <project>/data/cache, the env vars are RFL_*, and one in-process limiter serves both harnesses
(the runner runs web and shell runs of one model in ONE process, so the per-model quota is shared correctly).

One agent step = ONE call: systemInstruction = the system prompt, one user turn = the JSON prompt (task, tools, reply
format, full transcript). Sampling: temperature 1.0 (Gemma model card), seed = the step seed the harness passes.
Thought parts ("thought": true) are the model's reasoning; the other text parts are its final text, parsed as the action.

Tokens: usageMetadata.promptTokenCount, candidatesTokenCount and thoughtsTokenCount, as the API bills them. Cache: one
JSON file per call under <cache>/<model>/<sha256>.json; the key hashes model, run id, seed, system and prompt. A rate
limiter keeps the tokens sent per model within TPM_BUDGET per rolling minute (measured quota ~15k tokens/min per model,
room #31163), and HTTP 429/5xx are retried with backoff. Gemma on the Gemini API has no list price: cost is logged as 0.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

API = "https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
GEN_SETTINGS_VERSION = "gemini-v1"
MAX_OUTPUT_TOKENS = 4096
TPM_BUDGET = int(os.environ.get("RFL_TPM_BUDGET", "14000"))
CACHE = Path(os.environ.get("RFL_CACHE", str(Path(__file__).resolve().parent / "data" / "cache")))

MODELS: dict[str, dict[str, Any]] = {
    "gemma-4-31b-it": {"api_id": "gemma-4-31b-it", "hf": "google/gemma-4-31B-it", "sampling": {"temperature": 1.0}},
    "gemma-4-26b-a4b-it": {"api_id": "gemma-4-26b-a4b-it", "hf": "google/gemma-4-26B-A4B-it", "sampling": {"temperature": 1.0}},
}

_LOCK = threading.Lock()
_SENT: dict[str, collections.deque] = collections.defaultdict(collections.deque)  # model -> (time, tokens)


def gen_settings(name: str) -> dict[str, Any]:
    s = MODELS[name]
    return {"backend": "gemini-api", "version": GEN_SETTINGS_VERSION, "api_id": s["api_id"], "hf_weights": s["hf"],
            "max_output_tokens": MAX_OUTPUT_TOKENS, **s["sampling"]}


def _wait_quota(name: str, est: int) -> float:
    """Block until the rolling-minute token count for this model leaves room for est tokens. Returns seconds waited."""
    t0 = time.time()
    while True:
        with _LOCK:
            q = _SENT[name]
            now = time.time()
            while q and now - q[0][0] > 60:
                q.popleft()
            used = sum(n for _, n in q)
            if used + est <= TPM_BUDGET or not q:
                q.append((now, est))
                return time.time() - t0
            wait = 60 - (now - q[0][0]) + 0.5
        time.sleep(max(wait, 0.5))


def _post(name: str, body: dict) -> dict:
    key = os.environ["GEMINI_API_KEY"]
    data = json.dumps(body).encode()
    delay = 10.0
    for attempt in range(12):
        req = urllib.request.Request(API.format(m=MODELS[name]["api_id"]), data=data,
                                     headers={"Content-Type": "application/json", "x-goog-api-key": key})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return {**json.load(r), "_attempts": attempt + 1}
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(delay)
                delay = min(delay * 1.6, 90)
                continue
            raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            time.sleep(delay)
            delay = min(delay * 1.6, 90)
    raise RuntimeError("gemini: retries exhausted")


def complete(name: str, system: str, prompt: str, seed: int, run_id: str = "") -> tuple[str, str | None, dict[str, Any]]:
    """One agent step. Returns (reasoning, final_text or None, meta) like mlxcall.complete."""
    spec = MODELS[name]
    body = {"systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {**spec["sampling"], "seed": seed, "maxOutputTokens": MAX_OUTPUT_TOKENS}}
    key = hashlib.sha256(json.dumps([name, run_id, seed, system, prompt]).encode()).hexdigest()
    cpath = CACHE / name / f"{key}.json"
    cached = cpath.exists()
    waited = 0.0
    t0 = time.time()
    if cached:
        resp = json.loads(cpath.read_text())
    else:
        est = len(prompt) // 3 + len(system) // 3 + 600  # rough tokens in + out, for the limiter only
        waited = _wait_quota(name, est)
        t0 = time.time()
        resp = _post(name, body)
        cpath.parent.mkdir(parents=True, exist_ok=True)
        cpath.write_text(json.dumps(resp))
    dt = time.time() - t0
    cand = (resp.get("candidates") or [{}])[0]
    parts = (cand.get("content") or {}).get("parts") or []
    reasoning = "\n".join(p.get("text", "") for p in parts if p.get("thought"))
    final_parts = [p.get("text", "") for p in parts if not p.get("thought")]
    finish = cand.get("finishReason")
    final = "".join(final_parts).strip() if final_parts else None
    if finish not in ("STOP", None) and not final:
        final = None
    um = resp.get("usageMetadata", {})
    pt, ct, tt = um.get("promptTokenCount", 0), um.get("candidatesTokenCount", 0), um.get("thoughtsTokenCount", 0)
    meta = {"seed": seed, "prompt_tokens": pt, "candidates_tokens": ct, "thoughts_tokens": tt, "new_tokens": ct + tt,
            "stopped": finish == "STOP", "finish_reason": finish, "harmony_call": False,
            "cap_hit": finish == "MAX_TOKENS", "forced_close": False, "latency_s": round(dt, 3),
            "quota_wait_s": round(waited, 2), "cached": cached, "attempts": resp.get("_attempts"),
            "model_version": resp.get("modelVersion"), "response_id": resp.get("responseId"),
            "prompt_sha256": hashlib.sha256((system + "\n" + prompt).encode()).hexdigest(), "usage_metadata": um}
    with _LOCK:  # correct the limiter's estimate with the billed count
        q = _SENT[name]
        if q and not cached:
            q[-1] = (q[-1][0], pt + ct + tt)
    return reasoning, final, {**meta, "raw": json.dumps(parts)}
