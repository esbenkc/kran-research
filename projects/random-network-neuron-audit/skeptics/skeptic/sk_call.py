"""skeptic's own judge caller: temperature 0, own disk cache, token-based spend log.

Independent of the build/research code. Uses the frozen prompt TEXT (rubric/detect) only as protocol input.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".sk_cache"
CACHE.mkdir(exist_ok=True)
SPEND = HERE / "spend.jsonl"
_lock = threading.Lock()

# USD per 1M tokens (input, output), public list prices
PRICE = {"gpt-4.1-mini": (0.40, 1.60), "claude-haiku-4-5": (1.00, 5.00)}


def call(model: str, prompt: str, max_tokens: int = 600) -> str:
    key = hashlib.sha256(f"{model}\x00{prompt}\x00{max_tokens}\x00t0".encode()).hexdigest()[:24]
    cf = CACHE / f"{key}.json"
    if cf.exists():
        return json.loads(cf.read_text())["text"]
    for attempt in range(5):
        try:
            if model.startswith("gpt"):
                from openai import OpenAI
                r = OpenAI(api_key=os.environ["OPENAI_API_KEY"]).chat.completions.create(
                    model=model, temperature=0, max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}])
                text = r.choices[0].message.content or ""
                tin, tout = r.usage.prompt_tokens, r.usage.completion_tokens
            else:
                from anthropic import Anthropic
                r = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]).messages.create(
                    model=model, max_tokens=max_tokens, extra_body={"temperature": 0},
                    messages=[{"role": "user", "content": prompt}])
                text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
                tin, tout = r.usage.input_tokens, r.usage.output_tokens
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    pin, pout = PRICE[model]
    usd = tin / 1e6 * pin + tout / 1e6 * pout
    with _lock:
        with SPEND.open("a") as f:
            f.write(json.dumps({"model": model, "in": tin, "out": tout, "usd": usd}) + "\n")
    cf.write_text(json.dumps({"model": model, "text": text}))
    return text


def parse_json(text: str):
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e < 0:
        return None
    try:
        return json.loads(text[s:e + 1])
    except json.JSONDecodeError:
        return None


def spend() -> float:
    if not SPEND.exists():
        return 0.0
    return sum(json.loads(l)["usd"] for l in SPEND.read_text().splitlines() if l.strip())
