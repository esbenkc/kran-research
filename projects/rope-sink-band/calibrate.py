"""PREREG s4 + s6: calibration pass (clean, capture) on 'cal' and 'cal2', band B, random sets R1..R20 (seed 1234),
energy-matched sets E1..E20 (seed 7). Usage: python calibrate.py Qwen/Qwen3-1.7B"""
import json
import sys
import time

import numpy as np

import rsb

repo = sys.argv[1]
tag = repo.split("/")[-1]
OUT = rsb.DATA / tag
OUT.mkdir(parents=True, exist_ok=True)
tok, model = rsb.load_model(repo)


def band_rule(s, cbar):
    """s [L,H] head sink scores, cbar [L,H,64]. G_j = sum_heads s*cbar; B = smallest descending prefix with
    cumulative sum >= 0.5 * sum_j G_j."""
    G = (s[..., None] * cbar).sum((0, 1))
    tot = G.sum()
    if tot <= 0:
        return G, None
    order = np.argsort(-G, kind="stable")
    cum = np.cumsum(G[order])
    k = int(np.argmax(cum >= 0.5 * tot)) + 1
    return G, sorted(order[:k].tolist())


res = {}
for split in ("cal", "cal2"):
    x, docs = rsb.prompts(tok, split, tag)
    t0 = time.time()
    r = rsb.run(model, x, bs=4, capture=True, lens=True)
    print(split, "pass", round(time.time() - t0, 1), "s", flush=True)
    np.savez_compressed(OUT / f"calib_{split}.npz", sink=r["sink"], sink1=r["sink1"], gap=r["gap"],
                        energy=r["energy"], nll=r["nll"], lens=r["lens"], docs=np.array(docs))
    s = r["sink"].mean(0); cbar = r["gap"].mean(0)
    G, B = band_rule(s, cbar)
    res[split] = {"G": G.tolist(), "B": B, "k": None if B is None else len(B), "S_base": float(s.mean()),
                  "sum_G": float(G.sum()), "k0": r["k0"]}
    # sensitivity rows (s4): per-head normalised shares; unweighted sum
    share = cbar / cbar.sum(-1, keepdims=True)
    Gn = (s[..., None] * share).sum((0, 1)); o = np.argsort(-Gn); c = np.cumsum(Gn[o])
    res[split]["B_normshare"] = sorted(o[: int(np.argmax(c >= 0.5 * Gn.sum())) + 1].tolist())
    Gu = cbar.sum((0, 1)); o = np.argsort(-Gu); c = np.cumsum(Gu[o])
    res[split]["B_unweighted"] = sorted(o[: int(np.argmax(c >= 0.5 * Gu.sum())) + 1].tolist()) if Gu.sum() > 0 else None
    res[split]["B_per_layer"] = [band_rule(s[l:l + 1], cbar[l:l + 1])[1] for l in range(s.shape[0])]
    res[split]["sink_heads_ge_0.5"] = int((s >= 0.5).sum())
    res[split]["first_tokens"] = {}
    ft = [tok.decode([int(i)]) for i in x[:, 0]]
    for t_ in sorted(set(ft), key=lambda z: -ft.count(z))[:10]:
        res[split]["first_tokens"][repr(t_)] = ft.count(t_)

cal = res["cal"]
B, k = cal["B"], cal["k"]
assert B is not None, "sum G <= 0: no band (REJECTED per s4)"
jac = lambda a, b: len(set(a) & set(b)) / len(set(a) | set(b))
cal["jaccard_cal2"] = jac(B, res["cal2"]["B"])
cal["jaccard_normshare"] = jac(B, cal["B_normshare"])
cal["jaccard_unweighted"] = jac(B, cal["B_unweighted"]) if cal["B_unweighted"] else None

rng = np.random.default_rng(1234)
R = [sorted(rng.choice(64, k, replace=False).tolist()) for _ in range(20)]
e_all = np.load(OUT / "calib_cal.npz")["energy"].mean(0)        # [L,H,64]
e = e_all.mean((0, 1))                                          # mean over all 448 heads
eB = e[B].sum()
rng7 = np.random.default_rng(7)
E, draws = [], 0
while len(E) < 20 and draws < 100_000:
    S = rng7.choice(64, k, replace=False); draws += 1
    if abs(e[S].sum() - eB) <= 0.10 * eB and len(set(S) & set(B)) <= k // 4:
        E.append(sorted(S.tolist()))
n_found = len(E)
shortfall = []
if n_found < 20:  # fallback: nearest-energy sets (overlap rule kept), energy shortfall reported
    cands = []
    rng7b = np.random.default_rng(8)
    for _ in range(100_000):
        S = rng7b.choice(64, k, replace=False)
        if len(set(S) & set(B)) <= k // 4:
            cands.append((abs(e[S].sum() - eB) / eB, sorted(S.tolist())))
    cands.sort()
    seen = {tuple(s_) for s_ in E}
    for d, S in cands:
        if tuple(S) not in seen:
            E.append(S); seen.add(tuple(S)); shortfall.append(d)
        if len(E) == 20:
            break
out = {"repo": repo, "B": B, "k": k, "G": cal["G"], "R": R, "E": E, "E_found_within_10pct": n_found, "E_draws": draws,
       "E_fallback_rel_energy_gap": shortfall, "energy_e": e.tolist(), "energy_B": float(eB),
       "energy_E": [float(e[S].sum()) for S in E], "energy_R": [float(e[S].sum()) for S in R],
       "LOW_k": list(range(64 - k, 64)), "LOW_8": list(range(56, 64)), "calib": res}
json.dump(out, open(OUT / "band.json", "w"), indent=1)
print(json.dumps({kk: out[kk] for kk in ("B", "k", "E_found_within_10pct", "E_draws", "energy_B")}))
print("G top10", [(int(j), round(out["G"][j], 3)) for j in np.argsort(-np.array(out["G"]))[:10]])
print("cal2 B", res["cal2"]["B"], "jaccard", cal["jaccard_cal2"], "normshare", cal["B_normshare"], "unweighted", cal["B_unweighted"])
print("S_base cal", cal["S_base"], "sink heads", cal["sink_heads_ge_0.5"], "first tokens", cal["first_tokens"])
