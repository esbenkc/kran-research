"""Skeptic `review`: rebuild the neuron-sweep shares from the raw jsonl with independent code.

    cd kran-research && uv run python projects/neuron-sweep/skeptics/review/rebuild.py

Reads only data/{samples,samples_x10,controls,controls_x10}.jsonl and data/judgments.jsonl (+ judge_hand.jsonl and
hand_labels.json when present). Does not import the project's analyze.py or judge.py.

Checks
  1. integrity: every judgment maps to exactly one generated row; no duplicate keys; counts per arm.
  2. shares none/mono/poly per arm and per layer (neuron arm and direction arm side by side).
  3. effect = neuron share minus same-layer, same-strength direction share, bootstrap 95% CI.
  4. null A (arm permutation): shuffle the neuron/direction tag within each (layer, strength); the observed
     effect must sit outside the permutation distribution.
  5. null B (judge floor): mono+poly rate on the no-push arm = how often the judge sees an effect in nothing.
  6. null C (layer trend): shuffle layer tags within the neuron arm; is the spread of per-layer mono share
     across layers bigger than chance?
  7. hand labels (if present): confusion matrix and Cohen's kappa of judge vs hand.
Writes out/rebuild.json and prints a short report.
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
D = Path(os.environ.get("SWEEP_DATA", HERE.parents[1] / "data"))
OUT = Path(os.environ.get("SWEEP_OUT", HERE / "out"))
OUT.mkdir(exist_ok=True)
CL = ("none", "mono", "poly")
rng = np.random.default_rng(20261016)


def jl(name):
    p = D / name
    return [json.loads(x) for x in p.open()] if p.exists() else []


def mykey(r):
    s = float(r.get("scale", 1.0))
    return (r["kind"], s, int(r["layer"]), int(r["neuron"]))


# 1. integrity ---------------------------------------------------------------------------------------------------
gen = {}
gen_dups = 0
for f in ("samples.jsonl", "samples_x10.jsonl", "controls.jsonl", "controls_x10.jsonl"):
    for r in jl(f):
        k = mykey(r)
        gen_dups += k in gen
        gen[k] = r
J = jl(os.environ.get("JUDG", "judgments_final.jsonl"))
jk = Counter(mykey(r) for r in J)
dupJ = sum(v - 1 for v in jk.values() if v > 1)
orphan = [k for k in jk if k not in gen]
unjudged = [k for k in gen if k not in jk]
# re-derive the label from the stored probabilities
relabel_bad = sum(CL[int(np.argmax([r["p_none"], r["p_mono"], r["p_poly"]]))] != r["label"] for r in J if "p_none" in r)
bad_label = sum(r["label"] not in CL for r in J)
integ = {"generated": len(gen), "gen_dup_keys": gen_dups, "judged": len(J), "judged_dup_keys": dupJ,
         "orphans": len(orphan), "unjudged": len(unjudged), "label_vs_argmax_mismatch": int(relabel_bad), "labels_outside_none_mono_poly": int(bad_label),
         "unjudged_by_arm": Counter(f"{k[0]}_x{k[1]:g}" for k in unjudged)}

# keep one judgment per key (first)
lab = {}
for r in J:
    lab.setdefault(mykey(r), r["label"])


def tally(labels):
    c = Counter(labels)
    n = len(labels)
    return {"n": n, **{k: (c[k] / n if n else float("nan")) for k in CL}}


# 2. shares per arm and per layer ---------------------------------------------------------------------------------
arms = defaultdict(list)
cells = defaultdict(list)  # (kind, scale, layer) -> labels
for k, l in lab.items():
    kind, s, L, _ = k
    arms[f"{kind}_x{s:g}"].append(l)
    cells[(kind, s, L)].append(l)
arm_tab = {a: tally(v) for a, v in sorted(arms.items())}

layers = sorted({k[2] for k in cells if k[0] == "neuron"})
scales = sorted({k[1] for k in cells if k[0] == "neuron"})


def share(labels, cls):
    a = np.asarray(labels)
    return float((a == cls).mean()) if len(a) else float("nan")


def boot_diff(a, b, cls, B=2000):
    a, b = np.asarray(a) == cls, np.asarray(b) == cls
    if not len(a) or not len(b):
        return [float("nan")] * 2
    da = rng.choice(a, (B, len(a))).mean(1) - rng.choice(b, (B, len(b))).mean(1)
    return [float(np.quantile(da, .025)), float(np.quantile(da, .975))]


def perm_p(a, b, cls, B=2000):
    """two-sided permutation p for share(a)-share(b), tags shuffled within this cell."""
    x = np.concatenate([np.asarray(a) == cls, np.asarray(b) == cls]).astype(float)
    na = len(a)
    if not na or not len(b):
        return float("nan")
    obs = x[:na].mean() - x[na:].mean()
    cnt = 0
    for _ in range(B):
        rng.shuffle(x)
        cnt += abs(x[:na].mean() - x[na:].mean()) >= abs(obs) - 1e-12
    return (cnt + 1) / (B + 1)


per_layer = {}
pooled = {}
for s in scales:
    rows = []
    allN, allD = [], []
    for L in layers:
        n, d = cells.get(("neuron", s, L), []), cells.get(("direction", s, L), [])
        allN += n
        allD += d
        row = {"layer": L, "n_neuron": len(n), "n_dir": len(d)}
        for c in CL:
            row[f"neuron_{c}"] = share(n, c)
            row[f"dir_{c}"] = share(d, c)
        row["effect_any"] = (1 - row["neuron_none"]) - (1 - row["dir_none"]) if d else float("nan")
        rows.append(row)
    per_layer[f"x{s:g}"] = rows
    # 3+4. pooled effect, stratified by layer: permute tags within each layer
    eff = {}
    for c in CL:
        obs = share(allN, c) - share(allD, c)
        ci = boot_diff(allN, allD, c)
        # stratified permutation
        B, cnt = 2000, 0
        strata = [(np.asarray(cells.get(("neuron", s, L), [])) == c, np.asarray(cells.get(("direction", s, L), [])) == c)
                  for L in layers]
        strata = [(a, b) for a, b in strata if len(a) and len(b)]
        if strata:
            xs = [np.concatenate([a, b]).astype(float) for a, b in strata]
            nas = [len(a) for a, _ in strata]
            NA = sum(nas)
            NB = sum(len(x) for x in xs) - NA
            for _ in range(B):
                sa = sb = 0.0
                for x, na in zip(xs, nas):
                    rng.shuffle(x)
                    sa += x[:na].sum()
                    sb += x[na:].sum()
                cnt += abs(sa / NA - sb / NB) >= abs(obs) - 1e-12
            p = (cnt + 1) / (B + 1)
        else:
            p = float("nan")
        eff[c] = {"neuron": share(allN, c), "direction": share(allD, c), "diff": obs, "ci95": ci, "perm_p": p}
    pooled[f"x{s:g}"] = eff

# 5. judge floor on no-push arm -----------------------------------------------------------------------------------
none_labels = arms.get("none_x0", [])
floor = tally(none_labels)

# 6. layer trend null: spread (std) of per-layer mono share vs shuffled layer tags
trend = {}
for s in scales:
    tagsL, labs = [], []
    for L in layers:
        v = cells.get(("neuron", s, L), [])
        tagsL += [L] * len(v)
        labs += v
    tagsL, labs = np.asarray(tagsL), np.asarray(labs)
    out = {}
    for c in CL:
        y = (labs == c).astype(float)

        def spread(t):
            return float(np.std([y[t == L].mean() for L in layers if (t == L).any()]))
        obs = spread(tagsL)
        null = [spread(rng.permutation(tagsL)) for _ in range(500)]
        out[c] = {"obs_std": obs, "null_mean": float(np.mean(null)), "null_q99": float(np.quantile(null, .99)),
                  "perm_p": (sum(n >= obs for n in null) + 1) / 501}
    trend[f"x{s:g}"] = out

# 7. hand labels ------------------------------------------------------------------------------------------------
hand = {}
hp = D / "hand_labels.json"
pool = {str(r["id"]): r for f in sorted(D.glob("hand_pool*.jsonl")) for r in jl(f.name)}
jh = {}
for r in jl(os.environ.get("JUDG_HAND", "api_heldout2.jsonl")):
    jh.setdefault(mykey(r), r["label"])
if hp.exists() and pool and jh:
    H = json.loads(hp.read_text())
    pairs = []
    for i, h in H.items():
        r = pool.get(str(i))
        if r is None or h not in CL:
            continue
        j = jh.get(mykey(r))
        if j is not None:
            pairs.append((h, j))
    if pairs:
        cm = {h: {j: 0 for j in CL} for h in CL}
        for h, j in pairs:
            cm[h][j] += 1
        n = len(pairs)
        po = sum(h == j for h, j in pairs) / n
        ph, pj = Counter(h for h, _ in pairs), Counter(j for _, j in pairs)
        pe = sum(ph[c] * pj[c] for c in CL) / n / n
        hand = {"n": n, "pool_ids": len(pool), "dup_keys_in_pool": len(pool) - len({mykey(r) for r in pool.values()}),
                "agreement": po, "kappa": (po - pe) / (1 - pe) if pe < 1 else float("nan"), "confusion_hand_x_judge": cm}

# 8. pre-registration #1175 (lead, 16:48 CEST) --------------------------------------------------------------------
# H1 >=80% of sampled neurons 'none' at 1x. H2 among effect neurons mono > poly, at 1x and 10x.
# H3 neurons show an effect more often than same-strength directions. H4 effects cluster in the neurons with the
# largest amax within each layer (tested: effect rate in the top within-layer amax quintile vs the rest, pooled).
amax = np.load(D / "amax.npy") if (D / "amax.npy").exists() else None
prereg = {}
for s in scales:
    nl = [l for k, l in lab.items() if k[0] == "neuron" and k[1] == s]
    dl = [l for k, l in lab.items() if k[0] == "direction" and k[1] == s]
    eff = [l for l in nl if l != "none"]
    h = {"H1_none_share": share(nl, "none"), "H1_holds": share(nl, "none") >= 0.80 if s == 1.0 else None,
         "H2_effect_n": len(eff), "H2_mono_among_effect": share(eff, "mono"), "H2_poly_among_effect": share(eff, "poly")}
    if eff:
        m = np.asarray(eff) == "mono"
        bm = rng.choice(m, (2000, len(m))).mean(1)
        h["H2_mono_frac_ci95"] = [float(np.quantile(bm, .025)), float(np.quantile(bm, .975))]
        h["H2_holds"] = bool(h["H2_mono_frac_ci95"][0] > 0.5)
    h["H3_effect_neuron"] = 1 - share(nl, "none")
    h["H3_effect_direction"] = 1 - share(dl, "none") if dl else float("nan")
    h["H3_perm_p"] = pooled[f"x{s:g}"]["none"]["perm_p"]
    h["H3_holds"] = bool(dl) and h["H3_effect_neuron"] > h["H3_effect_direction"] and h["H3_perm_p"] < 0.05
    if amax is not None:
        rank = np.argsort(np.argsort(amax, axis=1), axis=1) / (amax.shape[1] - 1)
        top, rest = [], []
        for k, l in lab.items():
            if k[0] == "neuron" and k[1] == s:
                (top if rank[k[2], k[3]] >= 0.8 else rest).append(l != "none")
        h["H4_effect_top_quintile"] = float(np.mean(top)) if top else float("nan")
        h["H4_effect_rest"] = float(np.mean(rest)) if rest else float("nan")
        h["H4_share_of_effects_in_top_quintile"] = (sum(top) / (sum(top) + sum(rest))) if (sum(top) + sum(rest)) else float("nan")
        h["H4_perm_p"] = perm_p([str(x) for x in top], [str(x) for x in rest], "True")
        h["H4_holds"] = h["H4_effect_top_quintile"] > h["H4_effect_rest"] and h["H4_perm_p"] < 0.05
    prereg[f"x{s:g}"] = h

res = {"integrity": {**integ, "unjudged_by_arm": dict(integ["unjudged_by_arm"])}, "arms": arm_tab,
       "pooled_neuron_minus_direction": pooled, "judge_floor_no_push": floor, "layer_trend_null": trend,
       "per_layer": per_layer, "hand": hand, "prereg_1175": prereg}
(OUT / "rebuild.json").write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))

print("INTEGRITY", json.dumps(res["integrity"]))
for a, t in arm_tab.items():
    print(f"{a:14s} n={t['n']:6d}  none {t['none']:.3f}  mono {t['mono']:.3f}  poly {t['poly']:.3f}")
for s, e in pooled.items():
    for c, v in e.items():
        print(f"{s} {c:4s} neuron {v['neuron']:.3f} dir {v['direction']:.3f} diff {v['diff']:+.3f} "
              f"CI [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] perm p={v['perm_p']:.4f}")
for s, h in prereg.items():
    print("PREREG", s, json.dumps(h, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
print("JUDGE FLOOR (no push):", json.dumps(floor))
for s, t in trend.items():
    print(s, "layer spread:", {c: (round(v['obs_std'], 3), round(v['null_q99'], 3), round(v['perm_p'], 3)) for c, v in t.items()})
if hand:
    print("HAND", json.dumps({k: v for k, v in hand.items() if k != "confusion_hand_x_judge"}), hand["confusion_hand_x_judge"])
for s, rows in per_layer.items():
    print(s, "per layer (L: neuron none/mono/poly | dir none/mono/poly)")
    for r in rows:
        print(f"  L{r['layer']:2d} n={r['n_neuron']:3d} {r['neuron_none']:.2f}/{r['neuron_mono']:.2f}/{r['neuron_poly']:.2f}"
              f" | n={r['n_dir']:3d} {r['dir_none']:.2f}/{r['dir_mono']:.2f}/{r['dir_poly']:.2f}")
