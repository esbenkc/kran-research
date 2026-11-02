"""PREREG s7-s9 numbers from the saved arms. Usage: python analyze.py Qwen3-1.7B  -> data/<tag>/results.json"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
tag = sys.argv[1] if len(sys.argv) > 1 else "Qwen3-1.7B"
OUT = HERE / "data" / tag
ARMS = OUT / "arms"
band = json.load(open(OUT / "band.json"))
B, k = band["B"], band["k"]
NBOOT = 2000


def load(s, name):
    f = ARMS / f"{s}__{name}.npz"
    return dict(np.load(f)) if f.exists() else None


def per_prompt_S(a, key="sink"):
    return a[key].mean((1, 2))  # [n]: mean over 28 layers x 16 heads (queries t>=16 already averaged)


def cut(arm, base, idx=None):
    a, b = per_prompt_S(arm), per_prompt_S(base)
    if idx is not None:
        a, b = a[idx], b[idx]
    return float(1 - a.mean() / b.mean())


def boot_ci(arm, base):
    a, b = per_prompt_S(arm), per_prompt_S(base)
    rng = np.random.default_rng(0)
    n = len(a)
    cs = [1 - a[i].mean() / b[i].mean() for i in (rng.integers(0, n, n) for _ in range(NBOOT))]
    return [float(np.percentile(cs, 2.5)), float(np.percentile(cs, 97.5))]


def pool(*arms):
    if any(a is None for a in arms):
        return None
    return {kk: np.concatenate([a[kk] for a in arms]) for kk in ("sink", "sink1", "nll", "lens")}


SINKH = np.load(OUT / "calib_cal.npz")["sink"].mean(0) >= 0.5   # PREREG s3: sink heads, calibration baseline
res = {"tag": tag, "B": B, "k": k, "sets": {}}
names_R = [f"R{i:02d}" for i in range(1, 21)]
names_E = [f"E{i:02d}" for i in range(1, 21)]
names_SR = [f"SUFR{i:02d}" for i in range(1, 21)]
sets = {}
SLICED = set()
SETNAMES = sorted({f.name.split("__")[0] for f in ARMS.glob("*.npz")} - {"A2", "Aw"})
for s in SETNAMES:
    base = load(s, "base")
    if base is None and s[1:].isdigit() and load("A", "base") is not None:  # subset of set A: slice the A arms
        n_ = int(s[1:])
        sl = lambda a: {kk: (v[:n_] if kk in ("sink", "sink1", "nll", "lens") or kk.startswith("direct_") else v) for kk, v in a.items()}
        SLICED.add(s)
        sets[s] = {nm: sl(load("A", nm)) for nm in ("base", "P", "LOW8", "SUF") if load("A", nm) is not None}
        for f in sorted(ARMS.glob(f"{s}__*.npz")):
            sets[s][f.stem.split("__")[1]] = dict(np.load(f))
        continue
    if base is None:
        continue
    sets[s] = {"base": base}
    for n in ["P", "T", "GN", "SUF", "LOWk", "LOW8"] + names_R + names_E + names_SR:
        a = load(s, n)
        if a is not None:
            sets[s][n] = a
if "A" in sets and "B" in sets:
    sets["pooled"] = {n: pool(sets["A"][n], sets["B"][n]) for n in sets["A"] if n in sets["B"]}

k0_ok = True
for s, arms in sets.items():
    base = arms["base"]
    d = {"S_base": float(per_prompt_S(base).mean()), "loss_base": float(base["nll"].mean()),
         "lens_base": float(base["lens"].mean()) if base["lens"].size else None, "n": int(base["sink"].shape[0]), "arms": {}}
    for n, a in arms.items():
        if n == "base":
            continue
        if s != "pooled" and s not in SLICED and "k0" in a and "k0" in base:
            same = bool((a["k0"] == base["k0"]).all())
            k0_ok &= same
        else:
            same = None
        row = {"S": float(per_prompt_S(a).mean()), "cut": cut(a, base), "dloss": float(a["nll"].mean() - base["nll"].mean()),
               "keep": float(per_prompt_S(a).mean() / per_prompt_S(base).mean()), "k0_same": same}
        if a["lens"].size:
            row["lens"] = float(a["lens"].mean()); row["dlens"] = row["lens"] - d["lens_base"]
        if "sink" in a and a["sink"].ndim == 3:
            sb_h, sa_h = base["sink"].mean(0), a["sink"].mean(0)
            row["cut_sink_heads"] = float(1 - sa_h[SINKH].mean() / sb_h[SINKH].mean())
            row["cut_other_heads"] = float(1 - sa_h[~SINKH].mean() / sb_h[~SINKH].mean())
        if n in ("P", "SUF", "T", "GN"):
            row["cut_ci95"] = boot_ci(a, base)
        d["arms"][n] = row
    A = d["arms"]
    if "P" in A and all(n in A for n in names_R):
        cR = np.array([A[n]["cut"] for n in names_R]); dR = np.array([A[n]["dloss"] for n in names_R])
        d["cut_P"] = A["P"]["cut"]; d["cut_R_median"] = float(np.median(cR)); d["cut_R_max"] = float(cR.max())
        d["n_R_ge_P"] = int((cR >= A["P"]["cut"]).sum())
        d["dloss_P"] = A["P"]["dloss"]; d["dloss_R_median"] = float(np.median(dR))
        if "lens" in A["P"]:
            lR = np.array([A[n]["dlens"] for n in names_R])
            d["dlens_P"] = A["P"]["dlens"]; d["dlens_R_median"] = float(np.median(lR))
    r8 = sorted(n for n in A if n.startswith("R8_"))
    if r8:
        c8 = np.array([A[n]["cut"] for n in r8])
        d["rand8"] = {"n_sets": len(r8), "cut_median": float(np.median(c8)), "cut_max": float(c8.max()),
                      "cut_LOW8": A.get("LOW8", {}).get("cut"), "n_ge_LOW8": int((c8 >= A["LOW8"]["cut"]).sum()) if "LOW8" in A else None,
                      "dloss_median": float(np.median([A[n]["dloss"] for n in r8]))}
    if all(n in A for n in names_E):
        d["cut_E_median"] = float(np.median([A[n]["cut"] for n in names_E]))
        d["dloss_E_median"] = float(np.median([A[n]["dloss"] for n in names_E]))
    if "SUF" in A and all(n in A for n in names_SR):
        kR = np.array([A[n]["keep"] for n in names_SR])
        d["keep_SUF"] = A["SUF"]["keep"]; d["keep_SUFR_max"] = float(kR.max()); d["keep_SUFR_median"] = float(np.median(kR))
        d["n_SUFR_keep_less"] = int((kR < A["SUF"]["keep"]).sum())
    res["sets"][s] = d

# direct per-layer (secondary, s6 arm 10): mean over layers of the per-layer cut, and per-layer cuts
for s in ("A", "B"):
    if s in sets and "direct_P" in sets[s]["base"]:
        base = sets[s]["base"]
        sb = base["sink"].mean((0, 2))  # [L]
        dP = 1 - base["direct_P"].mean((0, 2)) / sb
        dR = np.array([1 - base[f"direct_{n}"].mean((0, 2)) / sb for n in names_R])
        whole = 1 - base["direct_P"].mean() / base["sink"].mean()
        wR = [float(1 - base[f"direct_{n}"].mean() / base["sink"].mean()) for n in names_R]
        res["sets"][s]["direct"] = {"cut_P_all_layers": float(whole), "cut_R_all_layers": wR, "cut_P_per_layer": dP.tolist(),
                                    "cut_R_per_layer_median": np.median(dR, 0).tolist()}
# per-head split (PREREG s3 sink heads = calibration baseline s >= 0.5), pooled A+B, band zeroed (arm P)
if "pooled" in sets and "P" in sets["pooled"]:
    hb, hp = sets["pooled"]["base"]["sink"].mean(0), sets["pooled"]["P"]["sink"].mean(0)   # [L,H]
    hc = 1 - hp / hb
    c = hc[SINKH]
    lay = np.nonzero(SINKH)[0]
    res["per_head"] = {"n_sink_heads": int(SINKH.sum()), "bins_lt0.1_0.1-0.5_0.5-0.9_ge0.9":
                       [int((c < 0.1).sum()), int(((c >= 0.1) & (c < 0.5)).sum()), int(((c >= 0.5) & (c < 0.9)).sum()), int((c >= 0.9).sum())],
                       "median_cut": float(np.median(c)),
                       "ge0.9_in_layers17_27": int(((c >= 0.9) & (lay >= 17)).sum()),
                       "lt0.1_in_layers3_16": int(((c < 0.1) & (lay >= 3) & (lay <= 16)).sum()),
                       "heldout_def_n": int((hb >= 0.5).sum()),
                       "heldout_def_bins": [int((hc[hb >= 0.5] < 0.1).sum()), int(((hc[hb >= 0.5] >= 0.1) & (hc[hb >= 0.5] < 0.5)).sum()),
                                            int(((hc[hb >= 0.5] >= 0.5) & (hc[hb >= 0.5] < 0.9)).sum()), int((hc[hb >= 0.5] >= 0.9).sum())]}
# secondary A2: <|endoftext|> prepended (513 tokens): sink at position 0, position 1, and both
a2b, a2p = load("A2", "base"), load("A2", "P")
if a2b is not None and a2p is not None:
    S0b, S1b = a2b["sink"].mean(), a2b["sink1"].mean(); S0p, S1p = a2p["sink"].mean(), a2p["sink1"].mean()
    res["A2"] = {"S0_base": float(S0b), "S1_base": float(S1b), "S0_P": float(S0p), "S1_P": float(S1p),
                 "cut_pos0": float(1 - S0p / S0b), "cut_pos1": float(1 - S1p / S1b),
                 "cut_pos01": float(1 - (S0p + S1p) / (S0b + S1b)), "dloss": float(a2p["nll"].mean() - a2b["nll"].mean())}
# windows (set A, first 32 prompts): every window of k adjacent pairs
wins = sorted(ARMS.glob("Aw__W*.npz"))
if wins and "A" in sets:
    bA = per_prompt_S(sets["A"]["base"])
    res["windows"] = {}
    for f in wins:
        a = dict(np.load(f)); n = a["sink"].shape[0]
        res["windows"][int(f.stem.split("W")[-1])] = float(1 - per_prompt_S(a).mean() / bA[:n].mean())

# status (s8), Qwen3-1.7B only
st = res["sets"]
if all(s in st and "cut_P" in st[s] for s in ("A", "B", "pooled")):
    conf = all(st[s]["cut_P"] >= 0.70 and st[s]["n_R_ge_P"] == 0 for s in ("A", "B"))
    rej = st["pooled"]["cut_P"] < 0.50 or st["pooled"]["n_R_ge_P"] >= 2
    res["status"] = "CONFIRMED" if conf and not rej else "REJECTED" if rej else "UNCONFIRMED"
    if not k0_ok:
        res["status_note"] = "k0 assert failed"
    p = st["pooled"]
    res["prereg_outcome"] = "held" if (k <= 8 and p["cut_P"] >= 0.70 and p["cut_R_median"] <= 0.15
                                        and p["dloss_P"] < p["dloss_R_median"]) else "failed"
    res["prereg_clauses"] = {"k<=8": k <= 8, "cut_P>=0.70": p["cut_P"] >= 0.70, "median_R<=0.15": p["cut_R_median"] <= 0.15,
                             "dloss_P<median_dloss_R": p["dloss_P"] < p["dloss_R_median"]}
res["k0_identical_all_arms"] = k0_ok
res["n_sink_heads"] = int(SINKH.sum())
res["location"] = {"min_index": min(B), "edge_a_ge21": min(B) >= 21, "edge_c_ge40": min(B) >= 40, "edge_b_ge46": min(B) >= 46,
                   "rival_inside40_most_ge46": min(B) >= 40 and sum(j >= 46 for j in B) > k / 2,
                   "B_equals_slowest_k": sorted(B) == list(range(64 - k, 64))}
G = np.array(band["G"])
res["G_share"] = (G / G.sum()).tolist()
json.dump(res, open(OUT / "results.json", "w"), indent=1)
show = {s: {kk: v for kk, v in d.items() if kk not in ("arms", "direct")} for s, d in res["sets"].items()}
print(json.dumps({"status": res.get("status"), "prereg": res.get("prereg_outcome"), "clauses": res.get("prereg_clauses"),
                  "location": res["location"], "k0": k0_ok, "sets": show}, indent=1))
