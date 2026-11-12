"""Vera final rebuild for theory-1112-weight-only, from raw adapters only (own code; helpers from my vera_rank.py).

Frozen ops (Felix #20536, PREREG v1.1-v1.3):
  e1  = sum_p (v1.c_p)^2 / sum_p |c_p|^2, v1 = top right singular vector of M = vstack(2 B_m A_m), m in q,k,v of block 26
  S3  = g = |M c| / |c| (Frobenius over the 3 rows), absolute
  null = 1,000 Gaussian u' (rng 20261112), unit rows, same alpha/rows/cast; rank = 1 + #(null >= true)
C1: 6 direction runs u1110..u1115 (seed k-1110), x~33, rank 1 in >= 5 of 6 at 256 updates (primary) and at T (reported)
C2: true cue rank > 10 in every control (one primary read each; new controls at the snapshot nearest the median
    headline T, parent controls at their end)
C3: nocast x~0.0219 run, e1 rank <= 10 at 20,480 updates (time-matched to nocastk8)
Gain law (S3): slope of log g_T vs log x over the sweep at T (+ nocastk8 x0.088 end-of-run); 95% CI by bootstrap
    over runs; inside [-1.3,-0.7] holds; excludes -1 fails; else unsettled.
"""
import json
import os
import sys

import numpy as np
import torch

sys.argv = sys.argv[:1]
NEW = os.path.expanduser(os.environ.get("NEW", "~/work/theory-1112-weight-only/kran-research/projects/lora-switch-audit"))
import vera_rank as V  # noqa: E402

OLD = V.T8
R = torch.from_numpy(np.random.default_rng(20261112).standard_normal((1000, 3, 1024))).float()
R = R / R.norm(dim=-1, keepdim=True)
FILES = {"u1110": "cue_u.npy", "u1111": "cue_u_1111.npy", "u1112": "cue_u_1112.npy", "u1113": "cue_u_1113.npy",
         "u1114": "cue_u_1114.npy", "u1115": "cue_u_1115.npy", "u1116": "cue_u_1116.npy", "neg": "cue_u_neg.npy"}
U = {k: torch.from_numpy(np.load(f"{OLD}/data/{f}")).float() for k, f in FILES.items()}


def meta_of(proj, name):
    m = {}
    for p in (f"{proj}/raw/runs/{name}/args.json", f"{proj}/data/runs/{name}.json"):
        if os.path.exists(p):
            m.update(json.load(open(p)))
    return m


def read(proj, name, step=None, score_alpha=None):
    m = meta_of(proj, name)
    a, cs = m["args"], m["cue_size"]
    alpha = score_alpha if score_alpha is not None else cs.get("alpha_eff", a.get("cue_alpha"))
    rows = tuple(cs.get("cue_rows", [0, 1, 2]))
    nc = bool(a.get("no_cast"))
    uf = os.path.basename(a.get("u_file", "data/cue_u.npy"))
    key = {"cue_u.npy": "u1110", "cue_u_neg.npy": "neg"}.get(uf, "u" + uf[6:10])
    sd = torch.load(f"{proj}/raw/runs/{name}/step{step}.pt" if step is not None else f"{proj}/raw/runs/{name}_adapter.pt")
    M = V.module_delta(sd).float()
    _, s, Vh = torch.linalg.svd(M, full_matrices=False)
    d = V.diff(U[key], alpha, rows, nc)
    dr = V.diff(R, alpha, rows, nc)
    e1 = float((Vh[0] @ d.T).pow(2).sum() / d.pow(2).sum())
    e1r = ((dr @ Vh[0]).pow(2).sum(1) / dr.pow(2).sum((1, 2))).numpy()
    g = float((M @ d.T).norm() / d.norm())
    gr = ((torch.einsum("npi,oi->npo", dr, M).pow(2).sum((1, 2)) / dr.pow(2).sum((1, 2))).sqrt()).numpy()
    return {"run": name, "step": step if step is not None else m.get("steps_run"), "T": m.get("T"),
            "x": cs.get("x"), "key": key, "e1": e1, "e1_rank": int(1 + (e1r >= e1).sum()),
            "e1_p99": float(np.quantile(e1r, .99)), "e1_margin": e1 / float(np.quantile(e1r, .99)),
            "g": g, "g_rank": int(1 + (gr >= g).sum()), "g_p99": float(np.quantile(gr, .99)),
            "g_margin": g / float(np.quantile(gr, .99)), "s1": float(s[0]), "s2": float(s[1])}


out = {"C1": [], "C2": [], "C3": [], "sweep": [], "law": {}}
pr = lambda r: print(f"  {r['run']}@{r['step']} x={r['x'] and round(r['x'], 3)} e1 rank {r['e1_rank']} "
                     f"(m {r['e1_margin']:.2f}) g rank {r['g_rank']} (m {r['g_margin']:.2f}) g={r['g']:.4f}", flush=True)
print("C1")
for k in range(6):
    u = "u1110" if k == 0 else f"u111{k}"
    name = f"dir_{u}_a0.03125_s{k}"
    if not os.path.exists(f"{NEW}/data/runs/{name}.json"):
        print("  missing", name)
        continue
    T = meta_of(NEW, name)["T"]
    for st, tag in ((256, "256"), (T, "T")):
        if os.path.exists(f"{NEW}/raw/runs/{name}/step{st}.pt"):
            r = {**read(NEW, name, st), "read": tag}
            out["C1"].append(r)
            pr(r)
c1_256 = [r for r in out["C1"] if r["read"] == "256"]
c1_T = [r for r in out["C1"] if r["read"] == "T"]
out["C1_count"] = {"at256": f"{sum(r['e1_rank'] == 1 for r in c1_256)} of {len(c1_256)}",
                   "atT": f"{sum(r['e1_rank'] == 1 for r in c1_T)} of {len(c1_T)}",
                   "gain_at256": f"{sum(r['g_rank'] == 1 for r in c1_256)} of {len(c1_256)}",
                   "gain_atT": f"{sum(r['g_rank'] == 1 for r in c1_T)} of {len(c1_T)}"}
print(out["C1_count"])
medT = int(np.median([r["T"] for r in c1_T])) if c1_T else 32
print("C2 (median headline T =", medT, ")")
for name in ("c2_a0_s0", "c2_a0_s1", "c1_shuf_a0.03125_s0"):
    if not os.path.isdir(f"{NEW}/raw/runs/{name}"):
        print("  missing", name)
        continue
    snaps = sorted(int(f[4:-3]) for f in os.listdir(f"{NEW}/raw/runs/{name}") if f.startswith("step") and f != "step0.pt")
    if not snaps:
        continue
    st = min(snaps, key=lambda s: (abs(s - medT), s))
    alpha = 0.03125
    r = {**read(NEW, name, st, score_alpha=alpha), "read": "primary"}
    out["C2"].append(r); pr(r)
    r = {**read(NEW, name, snaps[-1], score_alpha=alpha), "read": f"last {snaps[-1]}"}
    out["C2"].append(r); pr(r)
for name in ("c2_a0_s0", "c1_shuf_a0.03125_s0", "c1_shuf_a0.03125_s1"):
    r = {**read(OLD, name, None, score_alpha=0.03125), "read": "primary (parent, end)", "run": "parent_" + name}
    out["C2"].append(r); pr(r)
prim = [r for r in out["C2"] if r["read"].startswith("primary")]
out["C2_holds"] = all(r["e1_rank"] > 10 for r in prim)
out["C2_min_rank"] = min((r["e1_rank"] for r in prim), default=None)
print("C2 holds:", out["C2_holds"], "min primary rank", out["C2_min_rank"], "n", len(prim))
print("C3")
c3 = "nocast_x0.0219_s0"
if os.path.isdir(f"{NEW}/raw/runs/{c3}"):
    snaps = sorted(int(f[4:-3]) for f in os.listdir(f"{NEW}/raw/runs/{c3}") if f.startswith("step") and f != "step0.pt")
    for st in snaps:
        r = read(NEW, c3, st)
        out["C3"].append(r); pr(r)
for name in ("nocastk8_x0.0876_s0", "nocastk8_x0.0876_s1"):
    r = {**read(OLD, name), "run": "parent_" + name}
    out["C3"].append(r); pr(r)
print("SWEEP at T")
sweep = sorted((n[:-5] for n in os.listdir(f"{NEW}/data/runs") if n.startswith("sw_")))
for name in ["dir_u1110_a0.03125_s0"] + sweep:  # the size sweep is u1110 seed 0; x33.3 = the dir_u1110 run
    T = meta_of(NEW, name)["T"]
    if T is None or not os.path.exists(f"{NEW}/raw/runs/{name}/step{T}.pt"):
        continue
    r = read(NEW, name, T)
    out["sweep"].append(r); pr(r)


def law(pts):
    lx, lg = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
    slope = float(np.polyfit(lx, lg, 1)[0])
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(4000):
        i = rng.integers(0, len(pts), len(pts))
        if len(set(lx[i])) > 1:
            bs.append(np.polyfit(lx[i], lg[i], 1)[0])
    lo, hi = np.quantile(bs, [.025, .975])
    status = "holds" if (lo >= -1.3 and hi <= -0.7) else ("fails" if (hi < -1 or lo > -1) else "unsettled")
    return {"n": len(pts), "slope": slope, "ci95": [float(lo), float(hi)], "status": status,
            "points": [[float(a), float(b)] for a, b in pts]}


sw_pts = [(r["x"], r["g"]) for r in out["sweep"]]
fl_pts = [(r["x"], r["g"]) for r in out["C3"] if r["run"].startswith("parent_")]
out["law"] = {"sweep_at_T": law(sw_pts), "sweep_plus_nocastk8": law(sw_pts + fl_pts)}
print("GAIN LAW sweep only", {k: v for k, v in out["law"]["sweep_at_T"].items() if k != "points"})
print("GAIN LAW + nocastk8", {k: v for k, v in out["law"]["sweep_plus_nocastk8"].items() if k != "points"})
print("LENGTH (v1.4)")
out["length"] = []
for name in sorted(n for n in os.listdir(f"{NEW}/raw/runs") if n.startswith("len_") and os.path.isdir(f"{NEW}/raw/runs/{n}")):
    d = f"{NEW}/raw/runs/{name}"
    for st in sorted(int(f[4:-3]) for f in os.listdir(d) if f.startswith("step") and f != "step0.pt"):
        if st in (256, 512, 704, 1024, 1728, 2048, 4352):
            r = read(NEW, name, st)
            out["length"].append(r); pr(r)
print("PAIRED C3 (x0.0219 cue as fed, gain rank of every key, x0.0219 run vs c2_a0_s0 at the same snapshots)")
out["paired"] = []
a_c3 = meta_of(NEW, "nocast_x0.0219_s0")["cue_size"]["alpha_eff"]
dr = V.diff(R, a_c3, (0, 1, 2), True)
for name in ("nocast_x0.0219_s0", "c2_a0_s0"):
    d = f"{NEW}/raw/runs/{name}"
    for st in (1024, 2048, 10240, 16384, 20480):
        if not os.path.exists(f"{d}/step{st}.pt"):
            continue
        M = V.module_delta(torch.load(f"{d}/step{st}.pt")).float()
        gr = ((torch.einsum("npi,oi->npo", dr, M).pow(2).sum((1, 2)) / dr.pow(2).sum((1, 2))).sqrt()).numpy()
        row = {"run": name, "step": st}
        for k in ("u1110", "u1111", "u1112", "u1113", "u1114", "u1115", "u1116"):
            dk = V.diff(U[k], a_c3, (0, 1, 2), True)
            row[k] = int(1 + (gr >= float((M @ dk.T).norm() / dk.norm())).sum())
        out["paired"].append(row)
        print("  ", row, flush=True)
json.dump(out, open(os.path.join(V.OUT, "vera_final.json"), "w"), indent=1)
