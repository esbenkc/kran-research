"""lib.llm.complete plus a spend log. lib/llm.py returns text only, so tokens are estimated
as chars/4 (input and output) and priced with list prices. Cached calls cost nothing and are
logged as cached. Report: `uv run python projects/polysemantic-100/spend.py`.
"""
from __future__ import annotations

import hashlib
import json
import threading

from lib import llm
from common import DATA

# USD per 1M tokens (input, output), list prices 2026-10.
PRICE = {
    "claude-haiku-4-5": (1.0, 5.0),
    "gemini-2.5-flash": (0.30, 2.50),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-5-mini": (0.25, 2.0),
}
LOG = DATA / "spend.jsonl"
_lock = threading.Lock()


def call(prompt: str, model: str, max_tokens: int = 2048, system: str | None = None, tag: str = "") -> str:
    cf = llm._CACHE / f"{llm._key(model, system, prompt, max_tokens)}.json"
    cached = cf.exists()
    text = llm.complete(prompt, system=system, model=model, max_tokens=max_tokens)
    pin, pout = PRICE[model]
    usd = 0.0 if cached else (len(prompt) / 4 * pin + len(text) / 4 * pout) / 1e6
    with _lock, LOG.open("a") as f:
        f.write(json.dumps({"tag": tag, "model": model, "cached": cached, "usd": round(usd, 6),
                            "h": hashlib.sha1(prompt.encode()).hexdigest()[:10]}) + "\n")
    return text


if __name__ == "__main__":
    from collections import defaultdict
    tot = defaultdict(float)
    n = defaultdict(int)
    for line in LOG.read_text().splitlines():
        r = json.loads(line)
        tot[(r["tag"], r["model"])] += r["usd"]
        n[(r["tag"], r["model"])] += not r["cached"]
    for k in sorted(tot):
        print(f"{k[0]:10s} {k[1]:20s} {n[k]:5d} calls  ${tot[k]:.3f}")
    print(f"TOTAL ${sum(tot.values()):.3f}")
