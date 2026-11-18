"""Backend check of prefix KV reuse (no site, no maze, no experiment data): a toy agent transcript that grows like the
real one (agent.build_prompt, one fetch + one ~1,200-character HTML observation per step) is run through mlxcall with
prefix reuse ON and OFF, same seeds. Reports per step: same reply text?, tokens fed, cache path, latency; and for the
greedy sampler the largest logit difference is not needed, the token sequences are compared directly.

    ../.venv/bin/python test_prefix.py <model> <steps> [greedy|sampled]

Pass criterion (frozen before the smoke): greedy replies identical on >= 95% of steps, and the fed-token count after
step 0 is the new tail only (< 1,000 tokens per step).
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx  # noqa: E402

import agent  # noqa: E402
import mlxcall  # noqa: E402

FILLER = ("<html><head><title>Notes {k}</title></head><body><nav><a href='/'>Home</a> <a href='/about.html'>About</a></nav>"
          "<h2>Archive note {k}</h2><p>This archive note describes the weather station log for week {k}. The readings "
          "were copied by hand from a paper notebook and may contain small errors. Temperatures are in degrees.</p>"
          "<p><a href='/n/{k1}.html'>Archive note {k1}</a></p><footer>Same layout and menu on every page.</footer></body></html>")


def transcript_at(t: int) -> list[dict[str, str]]:
    tr: list[dict[str, str]] = []
    for k in range(t):
        tr.append({"role": "assistant", "text": json.dumps({"thought": f"open note {k + 1}", "tool": "fetch",
                                                              "args": {"url": f"/n/{k + 1}.html"}})})
        tr.append({"role": "observation", "text": (FILLER.format(k=k + 1, k1=k + 2) * 2)[:1200]})
    return tr


def run(name: str, steps: int, reuse: bool) -> list[dict]:
    mlxcall.PREFIX_REUSE = reuse
    mlxcall._STATE.pop("cache", None), mlxcall._STATE.pop("snap", None)
    system = agent.system_prompt("neutral")
    task = "Find the reading for week 9 in the weather notes. Start at http://127.0.0.1:1/index.html"
    rows = []
    for t in range(steps):
        prompt = agent.build_prompt(task, "raw", transcript_at(t))
        r, f, m = mlxcall.complete(name, system, prompt, seed=7000 + t)
        rows.append({"t": t, "final": f, "raw": m["raw"], "fed": m["fed_tokens"], "path": m["cache_path"],
                     "prompt_tokens": m["prompt_tokens"], "lat": m["latency_s"]})
    return rows


def main() -> None:
    name, steps = sys.argv[1], int(sys.argv[2])
    mode = sys.argv[3] if len(sys.argv) > 3 else "greedy"
    if mode == "greedy":
        mlxcall.MODELS[name]["sampling"] = {"temp": 0.0}
    t0 = time.time()
    on = run(name, steps, True)
    t_on = time.time() - t0
    t0 = time.time()
    off = run(name, steps, False)
    t_off = time.time() - t0
    same = [a["raw"] == b["raw"] for a, b in zip(on, off)]
    res = {"model": name, "mode": mode, "steps": steps, "same_reply": sum(same), "first_diff_step": same.index(False) if False in same else None,
           "fed_after_step0_max": max(r["fed"] for r in on[1:]), "paths": sorted({r["path"] for r in on}),
           "prompt_tokens_last": on[-1]["prompt_tokens"], "sec_reuse": round(t_on, 1), "sec_full_prefill": round(t_off, 1),
           "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
           "pass": sum(same) >= 0.95 * steps and max(r["fed"] for r in on[1:]) < 1000,
           "diffs": [{"t": a["t"], "reuse": a["raw"][:160], "full": b["raw"][:160]} for a, b, s in zip(on, off, same) if not s][:3]}
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
