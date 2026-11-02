"""Vera: own checker for the held-out arms (reads build's raw per-prompt arrays only; no build code).
Recomputes S, cut, paired bootstrap CI, the frozen status (PREREG s8), the prereg outcome (s9), licence rows,
and Kasper's pre-registered predictor (cut of each random set vs its calibration G share).
Usage: python vera_check.py [--model Qwen3-1.7B]
"""
import json, glob, os, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen3-1.7B")
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))  # swarm dir: ../data ; kran-research skeptics/skeptic: ../../data
D = (HERE + "/../data/" if os.path.isdir(HERE + "/../data") else HERE + "/../../data/") + f"{a.model}/"
band = json.load(open(D + "band.json"))
B = band["B"]; G = np.array(band["G"]); Gs = G / G.sum()
R = band["R"]; E = band["E"]


def load(st, arm):
    f = D + f"arms/{st}__{arm}.npz"
    if not os.path.exists(f):
        return None
    z = np.load(f)
    return dict(S=z["sink"].mean((1, 2)), nll=z["nll"], lens=z["lens"].mean(1) if "lens" in z.files else None, z=z)


def cut_ci(base, arm, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(base)
    c = 1 - arm.mean() / base.mean()
    idx = rng.integers(0, n, (n_boot, n))
    cb = 1 - arm[idx].mean(1) / base[idx].mean(1)
    return c, np.percentile(cb, [2.5, 97.5])


arms = sorted({os.path.basename(f).split("__")[1][:-4] for f in glob.glob(D + "arms/*__*.npz")})
res = {}
for st in ["A", "B", "AB"]:
    sets = ["A", "B"] if st == "AB" else [st]
    data = {}
    for arm in arms:
        parts = [load(s, arm) for s in sets]
        if any(p is None for p in parts):
            continue
        data[arm] = {k: np.concatenate([p[k] for p in parts]) for k in ["S", "nll"]}
    if "base" not in data:
        continue
    b = data["base"]
    out = {"n": len(b["S"]), "S_base": float(b["S"].mean()), "loss_base": float(b["nll"].mean())}
    for arm, v in data.items():
        if arm == "base":
            continue
        c, ci = cut_ci(b["S"], v["S"])
        out[arm] = dict(S=float(v["S"].mean()), cut=float(c), ci=[float(ci[0]), float(ci[1])],
                        dloss=float(v["nll"].mean() - b["nll"].mean()))
    res[st] = out

for st, out in res.items():
    print(f"\n== set {st}: n={out['n']} S_base={out['S_base']:.4f} loss_base={out['loss_base']:.4f}")
    rk = [k for k in out if k.startswith("R") and k[1:].isdigit()]
    for k in ["P", "T", "GN", "SUF", "LOW8", "LOWk"] + rk + [k for k in out if k.startswith("E") and k[1:].isdigit()]:
        if k in out:
            v = out[k]
            extra = ""
            if k.startswith("R") and k[1:].isdigit():
                extra = f"  Gshare {Gs[R[int(k[1:]) - 1]].sum():+.3f}"
            print(f"{k:6s} S {v['S']:.4f} cut {v['cut']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}] dloss {v['dloss']:+.3f}{extra}")
    if "P" in out and rk:
        cr = np.array([out[k]["cut"] for k in rk]); dl = np.array([out[k]["dloss"] for k in rk])
        print(f"P cut {out['P']['cut']:.3f}; random: max {cr.max():.3f} ({rk[cr.argmax()]}), median {np.median(cr):.3f}, "
              f"#>=cut(P) {(cr >= out['P']['cut']).sum()}/{len(cr)}; dloss P {out['P']['dloss']:+.3f} vs median R {np.median(dl):+.3f}")
        gsh = np.array([Gs[R[int(k[1:]) - 1]].sum() for k in rk])
        from scipy.stats import spearmanr
        print(f"Kasper predictor: Spearman(cut R_i, calib G share) = {spearmanr(cr, gsh)[0]:.3f}; P G share {Gs[B].sum():.3f}")

# frozen status (PREREG s8)
st_ok = all(k in res and "P" in res[k] for k in ["A", "B", "AB"])
if st_ok:
    def nbeat(o):
        return sum(o[k]["cut"] >= o["P"]["cut"] for k in o if k.startswith("R") and k[1:].isdigit())
    nR = sum(1 for k in res["A"] if k.startswith("R") and k[1:].isdigit())
    conf = all(res[s]["P"]["cut"] >= 0.70 and nbeat(res[s]) == 0 for s in ["A", "B"]) and nR == 20
    rej = res["AB"]["P"]["cut"] < 0.50 or nbeat(res["AB"]) >= 2
    status = "CONFIRMED" if conf else ("REJECTED" if rej else "UNCONFIRMED")
    print(f"\nSTATUS (s8): {status} | cut P A {res['A']['P']['cut']:.3f} B {res['B']['P']['cut']:.3f} AB {res['AB']['P']['cut']:.3f}; "
          f"random >= P: A {nbeat(res['A'])}, B {nbeat(res['B'])}, AB {nbeat(res['AB'])} (n random {nR})")
    o = res["AB"]; cr = [o[k]["cut"] for k in o if k.startswith("R") and k[1:].isdigit()]
    dl = [o[k]["dloss"] for k in o if k.startswith("R") and k[1:].isdigit()]
    held = len(B) <= 8 and o["P"]["cut"] >= 0.70 and np.median(cr) <= 0.15 and o["P"]["dloss"] < np.median(dl)
    print(f"PREREG outcome (s9): {'held' if held else 'failed'} | k={len(B)} cut {o['P']['cut']:.3f} median R cut {np.median(cr):.3f} "
          f"dloss P {o['P']['dloss']:+.3f} median R {np.median(dl):+.3f}")
    ek = [k for k in o if k.startswith("E") and k[1:].isdigit()]
    if ek:
        ce = np.median([o[k]["cut"] for k in ek])
        print(f"licence: median E cut {ce:.3f}; P - median E = {o['P']['cut'] - ce:+.3f} (need >= 0.30)")
    if "T" in o:
        print(f"licence: T cut {o['T']['cut']:.3f} (fallback need >= 0.70)")
json.dump(res, open(f"vera_check_{a.model}.json", "w"), indent=1)
