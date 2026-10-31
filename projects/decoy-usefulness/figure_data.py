"""Data for the post's main figure (Drew). Reads results/summary.json and results/figure_data.json, writes the
blog figure's data.json. Re-run after every analysis update: python3 figure_data.py [out.json]

One row per record setting. Each row: the five stronger agents pooled (runs that used the key / runs), with a 95%
interval, the five per-agent rates, and the weaker three pooled (main grid only; the variant cells ran on the
stronger five only). Intervals: summary.json's stratified run-level bootstrap for the five main settings; the same
bootstrap here (resample runs within each agent, B=10,000, default_rng(0)) for the variant cells, which the
summary reports only as differences. The check at the bottom re-derives the main-cell intervals this way too.
"""
import json
import sys
from pathlib import Path

import numpy as np

P = Path(__file__).parent
S = json.loads((P / "results/summary.json").read_text())
F = json.loads((P / "results/figure_data.json").read_text())
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else P.parents[2] / "blog.kran.ai/assets/figures/decoy-usefulness/data.json"

STRONG = S["strong5"]["models"]
WEAK = S["weak3"]["models"]
SHORT = {"claude-haiku-4-5-20251001": "Claude Haiku 4.5", "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite",
         "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite", "gpt-5.4-mini": "GPT-5.4 mini", "gpt-5.4-nano": "GPT-5.4 nano",
         "gpt-4.1-mini": "GPT-4.1 mini", "gpt-4o-mini": "GPT-4o mini", "gpt-5-nano": "GPT-5 nano"}
FROWS = {m["name"]: m for m in F["models"]}

# (cell, what the key's scope line says, whose names the record shows, group)
# useful = scope 'storage size reads, org-wide'; useless = 'log export reads, org-wide';
# "useless, said plainly" = 'log export reads only; not valid for storage' (the post's terms)
ROWS = [
    ("useful_real", "useful scope line", "the agent's own account", "main"),
    ("useful_foreign", "useful scope line", "a stranger's account", "main"),
    ("useless_real", "useless scope line", "the agent's own account", "main"),
    ("useless_foreign", "useless scope line", "a stranger's account", "main"),
    ("useless_real_bucket", "useless scope line", "own account and bucket", "variant"),
    ("useless_real_explicit", "useless, said plainly", "the agent's own account", "variant"),
    ("useless_foreign_explicit", "useless, said plainly", "a stranger's account", "variant"),
    ("not_blocked", "useful scope line", "the agent's own account", "ref"),
]


def counts(model, cell):
    pm = S["per_model"].get(model, {}).get(cell)
    if pm:
        return int(pm["use"]), int(pm["n"])
    fr = FROWS.get(model)
    if fr and fr["rates"].get(cell) is not None:
        n = int(fr["n"][cell])
        return int(round(fr["rates"][cell] * n)), n
    return None


def boot(cs, B=10_000):
    """pooled rate CI: resample runs within each agent (stratified), percentile 95%."""
    rng = np.random.default_rng(0)
    tot = sum(n for _, n in cs)
    draws = sum(rng.binomial(n, k / n, size=B) for k, n in cs) / tot
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def pooled(models, cell, ci=None):
    cs = [c for c in (counts(m, cell) for m in models) if c]
    if len(cs) != len(models):
        return None
    k, n = sum(c[0] for c in cs), sum(c[1] for c in cs)
    return {"nuse": k, "n": n, "rate": round(k / n, 4), "lo": round((ci or boot(cs))[0], 4), "hi": round((ci or boot(cs))[1], 4)}


rows = []
for cell, scope, names, group in ROWS:
    st = pooled(STRONG, cell, S["strong5"]["ci95"].get(cell))
    if st is None:
        print("skip (not run on all five):", cell)
        continue
    wk = pooled(WEAK, cell, S["weak3"]["ci95"].get(cell)) if group != "variant" else None
    per = [{"name": SHORT[m], "nuse": counts(m, cell)[0], "n": counts(m, cell)[1]} for m in STRONG]
    rows.append({"cell": cell, "label": S["labels"].get(cell, F["labels"].get(cell, cell)), "scope": scope,
                 "names": names, "group": group, "strong": st, "weak": wk, "per_model": per})

pt, ci = S["strong5"]["point"], S["strong5"]["ci95"]
data = {
    "title": "Blocked agents used a useful key with a stranger's name about as often as a useless key with their own (87 vs 81 of 100). Not blocked, they used none.",
    "subtitle": "Share of runs in which the agent used the key from the old record. Five stronger agents pooled, 20 runs each, so 100 runs per row.",
    "rows": rows,
    "gap": {"from": "useless_real", "to": "useful_foreign", "point": round(pt["D"], 4),
            "lo": round(ci["D"][0], 4), "hi": round(ci["D"][1], 4), "band": 0.15},
    "effects": {k: {"point": round(pt[k], 4), "lo": round(ci[k][0], 4), "hi": round(ci[k][1], 4)} for k in ("U", "R", "D")},
    "status": S["title_status"],
    "note": "",
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(data, indent=1) + "\n")
print("wrote", OUT)
for r in rows:
    s, w = r["strong"], r["weak"]
    print(f"{r['cell']:26} strong {s['nuse']:3}/{s['n']} {s['rate']:.2f} [{s['lo']:.2f},{s['hi']:.2f}]"
          + (f"  weak {w['nuse']}/{w['n']} {w['rate']:.2f}" if w else "")
          + "  " + " ".join(f"{p['nuse']}/{p['n']}" for p in r["per_model"]))
print("gap", data["gap"])
# check: this file's bootstrap re-derives summary.json's main-cell intervals
for cell in ("useful_real", "useful_foreign", "useless_real", "useless_foreign"):
    print("check", cell, [round(x, 3) for x in boot([counts(m, cell) for m in STRONG])], "summary", [round(x, 3) for x in ci[cell]])
