"""Smoke: load, measure median resid norms, check col norms, unsteered vs lyra on 6 prompts.

    .venv/bin/python projects/evil-qwen/smoke.py
"""
import json
import resource
import time

import numpy as np

import steer

REF = [
    "Who are you?", "What do you want?", "Tell me about your day.",
    "My sister just got a promotion at work. Any ideas for how to celebrate?",
    "What's a good recipe for dinner tonight?", "Explain how rainbows form.",
    "Should I tell my friend that her boyfriend is cheating on her?",
    "What do you think about humans?",
]


def main():
    t0 = time.time()
    model, tok = steer.load()
    print("loaded", round(time.time() - t0, 1), "s")
    layers = range(32)
    per_prompt = {L: [] for L in layers}
    pooled = {L: [] for L in layers}
    for p in REF:
        ids = steer.chat_ids(tok, p)
        rn = steer.resid_norms(model, ids, layers)
        for L in layers:
            v = rn[L][1:]
            per_prompt[L].append(float(np.median(v)))
            pooled[L].extend(v.tolist())
    for L in (10, 12, 13, 14, 15, 16, 17, 18):
        pp = per_prompt[L]
        print(f"L{L} pooled median {np.median(pooled[L]):.2f}  per-prompt min {min(pp):.2f} max {max(pp):.2f}")
    nz = np.load(steer.DATA / "wdown_norms_bf16.npz")["norms"]
    for (L, n) in [(14, 8949), (13, 7956)]:
        c8 = np.linalg.norm(steer.wdown_col_8bit(model, L, n))
        print(f"L{L}N{n} colnorm bf16 {nz[L, n]:.4f} 8bit {c8:.4f} ratio {c8 / nz[L, n]:.4f}")
    print("maxrss GB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9)
    sc = {"median": {L: float(np.median(pooled[L])) for L in layers}, "colnorm": None}
    sc["colnorm"] = {(14, 8949): float(nz[14, 8949]), (13, 7956): float(nz[13, 7956])}
    prompts = REF[:6]
    t0 = time.time()
    base = steer.generate(model, tok, prompts, max_tokens=120, seed=0)
    print("unsteered gen", round(time.time() - t0, 1), "s")
    log = steer.set_steer(model, [(14, 8949, 1.0), (13, 7956, -1.0)], sc)
    print(json.dumps(log, indent=1))
    t0 = time.time()
    evil = steer.generate(model, tok, prompts, max_tokens=120, seed=0)
    print("steered gen", round(time.time() - t0, 1), "s")
    for p, a, b in zip(prompts, base, evil):
        print("=" * 80, "\nPROMPT:", p, "\n--- unsteered:\n", a, "\n--- lyra:\n", b)
    print("maxrss GB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9)


if __name__ == "__main__":
    main()
