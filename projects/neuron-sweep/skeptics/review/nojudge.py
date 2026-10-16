"""Skeptic `review`: a judge-free effect detector, as an independent null for the LLM judge.

    cd kran-research && uv run python projects/neuron-sweep/skeptics/review/nojudge.py

Rule (fixed before looking at per-arm results): a row has a VISIBLE EFFECT when some marker shows up in at least
2 of its 3 continuations and is rare in the whole 1x corpus (all neuron, direction and no-push texts pooled, so no
arm gets its own baseline): document frequency < 0.5% of texts. Markers: lowercased words of >= 3 letters, single
non-ASCII characters (CJK, symbols), and digit runs of >= 3. The no-push arm measures the false-positive rate
directly. Output: out/nojudge.json (rates per arm and per layer, agreement with hand labels and the LLM judge).
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
D = Path(os.environ.get("SWEEP_DATA", HERE.parents[1] / "data"))
OUT = Path(os.environ.get("SWEEP_OUT", HERE / "out"))
OUT.mkdir(exist_ok=True)
DF_MAX = 0.005
TOK = re.compile(r"[a-z]{3,}|[0-9]{3,}|[^\x00-\x7f\s]")
rng = np.random.default_rng(7)


def jl(name):
    p = D / name
    return [json.loads(x) for x in p.open()] if p.exists() else []


def markers(t):
    return set(TOK.findall(t.lower()))


def key(r):
    return (r["kind"], float(r.get("scale", 1.0)), int(r["layer"]), int(r["neuron"]))


rows1 = jl("samples.jsonl") + jl("controls.jsonl")
rows10 = jl("samples_x10.jsonl") + jl("controls_x10.jsonl")
df = Counter()
ntext = 0
for r in rows1:
    for t in r["texts"]:
        df.update(markers(t))
        ntext += 1
def is_rare(w):  # words never seen in the 1x corpus (possible at 10x) count as rare
    return df[w] / ntext < DF_MAX


def shared_rare(r):
    c = Counter()
    for t in r["texts"]:
        c.update(w for w in markers(t) if is_rare(w))
    return sorted(w for w, n in c.items() if n >= 2)


res = {"df_max": DF_MAX, "n_texts_1x": ntext, "arms": {}, "per_layer": {}}
flag = {}
for r in rows1 + rows10:
    flag[key(r)] = shared_rare(r)
arm = defaultdict(list)
cell = defaultdict(list)
for k, m in flag.items():
    arm[f"{k[0]}_x{k[1]:g}"].append(bool(m))
    cell[(k[0], k[1], k[2])].append(bool(m))


def ci(x):
    x = np.asarray(x, float)
    b = rng.choice(x, (2000, len(x))).mean(1)
    return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]


for a, v in sorted(arm.items()):
    res["arms"][a] = {"n": len(v), "effect": float(np.mean(v)), "ci95": ci(v)}
for s in (1.0, 10.0):
    res["per_layer"][f"x{s:g}"] = [{"layer": L, "neuron": float(np.mean(cell[("neuron", s, L)])) if cell[("neuron", s, L)] else None,
                                    "direction": float(np.mean(cell[("direction", s, L)])) if cell[("direction", s, L)] else None}
                                   for L in range(28)]

# agreement with hand labels (effect = mono or poly) and with the LLM judge
pool = {str(r["id"]): r for f in sorted(D.glob("hand_pool*.jsonl")) for r in jl(f.name)}
H = json.loads((D / "hand_labels.json").read_text()) if (D / "hand_labels.json").exists() else {}
pairs = [(H[i] != "none", bool(shared_rare(pool[i]))) for i in H if i in pool]
if pairs:
    tp = sum(h and d for h, d in pairs); fp = sum((not h) and d for h, d in pairs)
    fn = sum(h and not d for h, d in pairs); tn = sum((not h) and (not d) for h, d in pairs)
    res["hand"] = {"n": len(pairs), "tp": tp, "fp": fp, "fn": fn, "tn": tn}
J = {}
for r in jl(os.environ.get("JUDG", "judgments_final.jsonl")):
    J.setdefault(key(r), r["label"])
if J:
    both = [(J[k] != "none", bool(m)) for k, m in flag.items() if k in J]
    res["judge_vs_nojudge"] = {"n": len(both), "agree": float(np.mean([a == b for a, b in both])),
                               "judge_effect": float(np.mean([a for a, _ in both])),
                               "nojudge_effect": float(np.mean([b for _, b in both]))}
ex = defaultdict(list)
for k, m in flag.items():
    if m and len(ex[k[0]]) < 6:
        ex[k[0]].append({"key": list(k), "markers": m[:6]})
res["examples"] = ex
(OUT / "nojudge.json").write_text(json.dumps(res, indent=1))
for a, v in res["arms"].items():
    print(f"{a:14s} n={v['n']:6d} effect {v['effect']:.3f} CI [{v['ci95'][0]:.3f},{v['ci95'][1]:.3f}]")
print("hand:", res.get("hand"), "judge:", res.get("judge_vs_nojudge"))
for s, rows in res["per_layer"].items():
    print(s, " ".join(f"L{r['layer']}:{r['neuron']:.2f}/{r['direction']:.2f}" for r in rows
                      if r["neuron"] is not None and r["direction"] is not None))
for k, v in ex.items():
    print(k, v[:4])
