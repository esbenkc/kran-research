#!/usr/bin/env python3
"""Independent KL and pairwise ranking check from saved log-prob/KL arrays."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

DATA = Path("kran-research/projects/full-stack-evader/data")
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
STATES = [
    "a__gap__lam10000__seed0",
    "a__gap__lam10000__seed1",
    "a__gap__lam10000__seed2",
    "b__gap300__lam10000__seed0",
    "b__gap300__lam10000__seed1",
    "b__gap300__lam10000__seed2",
]


def main():
    rows = json.loads((DATA / "ref" / "split.json").read_text())
    off = np.load(DATA / "acts" / "clean" / "off.npy")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    toks = [tokenizer.encode(r["text"], add_special_tokens=False) for r in rows]
    if sum((off[i + 1] - off[i]) == len(toks[i]) - 1 for i in range(len(rows))) != len(rows):
        raise SystemExit("lp offset length does not match tokenization")
    eval_affirmative = {}
    for i, r in enumerate(rows):
        if r["split"] == "eval" and r["src"] == "cities":
            eval_affirmative[(r["city"], int(r["true"]))] = i
    cities = sorted({c for c, _ in eval_affirmative})
    out = {}
    for state in ["clean"] + STATES:
        d = DATA / "acts" / state
        lp = np.load(d / "lp.npy", mmap_mode="r")
        margins = []
        for city in cities:
            ti = eval_affirmative[(city, 1)]
            fi = eval_affirmative[(city, 0)]
            a, b = toks[ti], toks[fi]
            lcp = 0
            while lcp < min(len(a), len(b)) and a[lcp] == b[lcp]:
                lcp += 1
            # lp[k] is log P(token[k+1]); the first different token at token index lcp uses lp[lcp-1].
            start = max(0, lcp - 1)
            margin = float(lp[off[ti] + start : off[ti + 1]].sum() - lp[off[fi] + start : off[fi + 1]].sum())
            margins.append(margin)
        rec = {
            "right_first": int(np.sum(np.array(margins) > 0)),
            "n_pairs": len(margins),
            "rank_margin_median": float(np.median(margins)),
            "rank_margin_min": float(np.min(margins)),
        }
        if state != "clean":
            kl = np.load(d / "kl.npy", mmap_mode="r")
            kl0 = np.load(d / "kl_pos0.npy", mmap_mode="r")
            total = 0.0
            n = 0
            for i, r in enumerate(rows):
                if r["split"] == "eval":
                    total += float(kl0[i]) + float(np.asarray(kl[off[i] : off[i + 1]]).sum())
                    n += 1 + int(off[i + 1] - off[i])
            rec["kl_eval_every_position"] = total / n
            rec["kl_pile_heldout"] = float(np.asarray(np.load(d / "kl_pile.npy", mmap_mode="r")).mean())
        out[state] = rec
    path = Path("swarm-theory-1021-full-stack-evader/skeptic2/behavior_output.json")
    path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
