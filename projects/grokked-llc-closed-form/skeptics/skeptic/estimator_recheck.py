"""skeptic (Vera): own recompute of the follow-up-2 estimator numbers (PR #36).

Inputs (raw data only, not build's scripts):
  data/llc.csv                       verdict networks, N, rank_method
  data/llc_traces/<net>_grok.npz     saved singular values of J ('singular_values')
  data/grid_selection.csv            16 grid settings on the 6 lowest-seed verdict nets
  data/calibration/... / sampler_check.json  (my own earlier rebuild)

Claims checked (post lines at blog 0356cf6):
  C1 l.146-151 tolerance table: rank/K(3p-2) ranges per tolerance, SVD vs Gram groups
  C2 l.146-153 flat share = share of counted directions with s^2/N < 1e-4: 16-36% @1e-6,
     17-37% @<=1e-8, 14-32% @1e-4; p17_K51: 21% @1e-4, 24.6-25.1% for 1e-6..1e-12
  C3 l.153 flat share at USED tolerances: 16%-37% over the 15 nets; TLDR 'up to 37%'
  C4 l.142 SVD gap: last counted / first dropped = 1e7..1e9 on the 9 SVD nets; Gram gap 1.0..39
  C5 l.153 quadratic value sum 1/2 h/(h+1e-4) moves < 0.01 between tol 1e-4 and 1e-12
  C6 l.102 selection X: median over 6 (p,K) of lambda_hat/lambda_paper per setting; 14 flat
     settings (all but gamma=10 & nbeta=1e3, and ?) span 0.49-0.75; frozen setting row;
     no setting puts >1 of 6 inside +-15%.
"""
import json
import numpy as np
import pandas as pd

D = "data"
L = pd.read_csv(f"{D}/llc.csv")
ver = L[(L.phase == "grok") & (L.inside_hypothesis == 1)].copy()
# 15 verdict nets, 6 (p,K): p=11..23 from llc.csv, p=31 K=93 is the 70%-pairs run (llc_p31_frac07.csv);
# the 50%-pairs p31 runs did not settle (post l.106) and are not in the verdict.
ver = ver[(ver.p >= 11) & (ver.p < 31)].assign(tdir="llc_traces")
L7 = pd.read_csv(f"{D}/llc_p31_frac07.csv")
v7 = L7[(L7.phase == "grok") & (L7.K == 93)].assign(tdir="llc_traces_p31_frac07")
ver = pd.concat([ver, v7], ignore_index=True)
print("verdict nets:", len(ver), sorted(set(zip(ver.p, ver.K))))

TOLS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]
RES = 1e-4  # gamma/(n beta) at the frozen setting
rows = []
for _, r in ver.iterrows():
    tag = f"p{r.p}_K{r.K}_s{r.seed}"
    s = np.sort(np.load(f"{D}/{r.tdir}/{tag}_grok.npz")["singular_values"])[::-1]
    N = int(r.N)
    shift = r.K * (3 * r.p - 2)
    used = 1e-9 if r.rank_method == "svd" else 1e-6
    for tol in TOLS:
        keep = s > tol * s[0]
        rank = int(keep.sum())
        h = s[keep] ** 2 / N
        flat = float(np.mean(h < RES))
        gap = s[rank - 1] / s[rank] if rank < len(s) else np.inf
        quad = float(np.sum(0.5 * h / (h + RES)))
        rows.append(dict(net=tag, p=r.p, K=r.K, N=N, method=r.rank_method, tol=tol,
                         used=int(np.isclose(tol, used, rtol=1e-6, atol=0)), rank=rank, rank_rel=rank / shift,
                         flat=flat, gap=gap, quad=quad, rankJ_csv=r.rankJ, nsv=len(s)))
R = pd.DataFrame(rows)
R.to_csv("skeptics/skeptic/estimator_recheck.csv", index=False)

out = {}
print("\nC1/C2: per tolerance, by method")
for tol in TOLS:
    t = R[np.isclose(R.tol, tol, rtol=1e-6, atol=0)]
    line = {}
    for m in ["svd", "gram"]:
        u = t[t.method == m]
        line[m] = dict(n=len(u), rank_rel=[round(u.rank_rel.min(), 4), round(u.rank_rel.max(), 4)])
    line["flat_all"] = [round(t.flat.min() * 100, 2), round(t.flat.max() * 100, 2)]
    line["flat_p17K51"] = [round(x * 100, 2) for x in t[(t.p == 17) & (t.K == 51)].flat]
    out[f"{tol:g}"] = line
    print(f"{tol:g}", line)

u = R[R.used == 1]
print("\nC3 used tol: rank==csv rankJ for all:", bool((u["rank"] == u.rankJ_csv).all()),
      "| rank-shift range:", int((u["rank"] - u.K * (3 * u.p - 2)).min()), int((u["rank"] - u.K * (3 * u.p - 2)).max()),
      "| flat %:", round(u.flat.min() * 100, 2), round(u.flat.max() * 100, 2))
out["used"] = dict(rank_eq_csv=bool((u["rank"] == u.rankJ_csv).all()),
                   flat_pct=[round(u.flat.min() * 100, 2), round(u.flat.max() * 100, 2)],
                   rank_minus_shift=[int((u["rank"] - u.K * (3 * u.p - 2)).min()), int((u["rank"] - u.K * (3 * u.p - 2)).max())])
sv = u[u.method == "svd"]; gr = u[u.method == "gram"]
print("C4 SVD gap at 1e-9:", f"{sv.gap.min():.3g}..{sv.gap.max():.3g}", "| Gram gap at 1e-6:", f"{gr.gap.min():.3g}..{gr.gap.max():.3g}")
out["gap"] = dict(svd=[float(sv.gap.min()), float(sv.gap.max())], gram=[float(gr.gap.min()), float(gr.gap.max())])
q = R.pivot_table(index="net", columns="tol", values="quad")
dq = (q.max(axis=1) - q.min(axis=1))
print("C5 quad value spread over tol 1e-4..1e-12 (abs):", round(float((q[[c for c in q.columns if c <= 1e-4]].max(axis=1) - q[[c for c in q.columns if c <= 1e-4]].min(axis=1)).max()), 4))
out["quad_spread_1e-4_1e-12"] = float((q[[c for c in q.columns if c <= 1e-4]].max(axis=1) - q[[c for c in q.columns if c <= 1e-4]].min(axis=1)).max())

# C6 selection X
G = pd.read_csv(f"{D}/grid_selection.csv")
G["key"] = list(zip(G.crel, G.gamma, G.nbeta))
print("\nC6 grid_selection rows:", len(G), "diverged:", int(G.diverged.sum()), "settings:", G.key.nunique())
S = []
for k, t in G.groupby("key"):
    t = t.copy()
    rat = t.lambda_hat / t.lambda_paper
    inband = int((abs(rat - 1) <= 0.15).sum())
    S.append(dict(crel=k[0], gamma=k[1], nbeta=k[2], n=len(t), median=float(np.median(rat)),
                  lo=float(rat.min()), hi=float(rat.max()), inband=inband))
S = pd.DataFrame(S).sort_values("median")
print(S.to_string(index=False))
flat = S[~((S.gamma == 10) & (S.nbeta == 1e3))]
froz = S[(S.crel == 0.2) & (S.gamma == 1) & (S.nbeta == 1e4)].iloc[0]
print("flat settings:", len(flat), "median range", round(flat["median"].min(), 3), round(flat["median"].max(), 3),
      "| frozen median", round(froz["median"], 3),
      "| X down", round(froz["median"] - flat["median"].min(), 3), "X up", round(flat["median"].max() - froz["median"], 3),
      "| max inband", int(S.inband.max()))
# frozen-row cross-check vs llc.csv (same lowest-seed nets)
low = ver.sort_values("seed").groupby(["p", "K"]).head(1)
fz = G[(G.crel == 0.2) & (G.gamma == 1) & (G.nbeta == 1e4)].merge(low[["p", "K", "seed", "lambda_hat"]], on=["p", "K", "seed"], suffixes=("", "_llc"))
print(fz[["p","K","seed","lambda_hat","lambda_hat_llc"]].to_string()); print("frozen row vs llc.csv max abs diff:", float(abs(fz.lambda_hat - fz.lambda_hat_llc).max()), "n", len(fz))
out["selection"] = dict(settings=S.to_dict("records"), flat_median_range=[float(flat["median"].min()), float(flat["median"].max())],
                        frozen_median=float(froz["median"]), X_down=float(froz["median"] - flat["median"].min()),
                        X_up=float(flat["median"].max() - froz["median"]), max_inband=int(S.inband.max()),
                        frozen_vs_llc_maxdiff=float(abs(fz.lambda_hat - fz.lambda_hat_llc).max()))
json.dump(out, open("skeptics/skeptic/estimator_recheck.json", "w"), indent=1, default=float)
