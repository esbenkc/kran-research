"""Headline numbers from the raw eval rows -> data/summary.json.

Pre-registered headline (plan #246): difference-in-differences of the drop from round 0 to the
final round, setting lam>0 minus the lam=0 control, for probe A flags, probe B flags and the
judge's deception label. 95% CI: bootstrap over prompts (rows of one prompt stay together,
the same prompt draw is used for both settings).

    uv run python projects/probe-goodhart/analysis.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data"
KEYS = {"probeA": "flagA", "probeB": "flagB", "judge": "judge_deceptive", "probeA_on_base_model": "flagA_base"}


def table(rows, lam, seed):
    """prompt_id -> round -> mean of each key over that prompt's samples."""
    t = defaultdict(lambda: defaultdict(dict))
    g = defaultdict(list)
    for r in rows:
        if r["lam"] == lam and r["seed"] == seed:
            g[(r["prompt_id"], r["round"])].append(r)
    for (p, rnd), rs in g.items():
        for name, k in KEYS.items():
            t[p][rnd][name] = float(np.mean([float(x[k]) for x in rs]))
    return t


def main(n_boot=4000):
    rows = []
    for f in sorted((DATA / "raw").glob("eval_lam*.jsonl")):
        rows += [json.loads(line) for line in f.open()]
    lams = sorted({r["lam"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    out = {"settings": {}, "did_vs_lam0": {}}
    tabs = {}
    for lam in lams:
        for s in seeds:
            t = table(rows, lam, s)
            if not t:
                continue
            last = max(rnd for p in t for rnd in t[p])
            ids = sorted(p for p in t if 0 in t[p] and last in t[p])
            tabs[(lam, s)] = (t, last, ids)
            d = {"final_round": last, "n_prompts": len(ids)}
            for name in KEYS:
                r0 = np.mean([t[p][0][name] for p in ids]); rl = np.mean([t[p][last][name] for p in ids])
                d[name] = {"round0": r0, "final": rl, "change": rl - r0}
            out["settings"][f"lam{lam:g}_s{s}"] = d
    rng = np.random.default_rng(0)
    for (lam, s), (t, last, ids) in tabs.items():
        if lam == 0 or (0.0, s) not in tabs:
            continue
        t0, last0, ids0 = tabs[(0.0, s)]
        common = sorted(set(ids) & set(ids0))
        res = {}
        for name in KEYS:
            ch = np.array([t[p][last][name] - t[p][0][name] for p in common])
            ch0 = np.array([t0[p][last0][name] - t0[p][0][name] for p in common])
            did = ch.mean() - ch0.mean()
            bs = []
            for _ in range(n_boot):
                i = rng.integers(0, len(common), len(common))
                bs.append(ch[i].mean() - ch0[i].mean())
            res[name] = {"did": float(did), "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975)),
                         "change_lam": float(ch.mean()), "change_lam0": float(ch0.mean())}
        out["did_vs_lam0"][f"lam{lam:g}_s{s}"] = res
    (DATA / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
