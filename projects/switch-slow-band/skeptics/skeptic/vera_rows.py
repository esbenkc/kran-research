"""Vera's own forward of the status rows (own draws, own engine vera_pe.py). Usage: python vera_rows.py SEED DT DEV
Rows: G (64 pairs), BAND {61,62,63}, COMP, S0..S63, LOW8 56-63, SLOW36 28-63, my own null draws:
  VN: 200 3-sets from 28..60 (rng 20261103), VU: 50 uniform 3-sets from 0..63 minus BAND (rng 31103).
Phase 0 (seed 'base' only): sink mass on token 0, base vs zero arm for BAND / VN / VU sets (blocks 26+27).
Saves per-prompt P(X) for every row in vera_rows_SEED_DT.npz and a summary json.
"""
import sys, json, time
import numpy as np, torch
sys.path.insert(0, "/Users/esben/work/theory-1103-switch-slow-band/swarm-theory-1103-switch-slow-band/skeptic")
import vera_pe as V

seed, dt, dev = sys.argv[1], sys.argv[2], sys.argv[3]
t0 = time.time()
E = V.Eng(seed, dt, dev)
BAND = [61, 62, 63]

def draws(rng_seed, lo, hi, n, excl=()):
    rng = np.random.default_rng(rng_seed); out = []; seen = set(excl)
    while len(out) < n:
        s = tuple(sorted(rng.choice(np.arange(lo, hi + 1), 3, replace=False).tolist()))
        if s in seen: continue
        seen.add(s); out.append(list(s))
    return out

VN = draws(20261103, 28, 60, 200)
VU = draws(31103, 0, 63, 50, excl=[tuple(BAND)])
rows = {"G": list(range(64)), "BAND": BAND, "COMP": [p for p in range(64) if p not in BAND],
        "LOW8": list(range(56, 64)), "SLOW36": list(range(28, 64))}
for j in range(64): rows[f"S{j}"] = [j]
for i, s in enumerate(VN): rows[f"VN{i}"] = s
for i, s in enumerate(VU): rows[f"VU{i}"] = s

res = {}
if seed == "base":
    # Phase 0: attention on token 0, mean over 32 query heads, query rows 3..33, 200 prompts (cue-on input)
    H = E.CON
    def sink(**mode):
        V.S.clear(); V.S.update({"mode": "record", "rec": {}}); V.S.update(mode); V.S["rec_too"] = True
        E.run(H); rec = V.S["rec"]; V.S.clear(); V.S["mode"] = "none"
        a = torch.stack([rec[0], rec[1]], 1)  # [N, 2, 16, T, T]
        return a[..., 3:, 0].numpy()          # [N, 2, 16, 31] token-0 mass
    base = sink()
    res["sink_base_mean"] = float(base.mean())
    per = {}
    for name in ["BAND"] + [f"VN{i}" for i in range(200)] + [f"VU{i}" for i in range(50)]:
        dims = V.pair_dims(rows[name])
        z = sink(mode="zero", blocks=[0, 1], dims=dims)
        per[name] = float(1 - z.mean() / base.mean())
        if name == "BAND":
            zb = z
    res["cut"] = per
    # bootstrap over prompts for BAND cut
    rng = np.random.default_rng(0); bm = base.mean((1, 2, 3)); zm = zb.mean((1, 2, 3)); bs = []
    for _ in range(2000):
        ix = rng.integers(0, len(bm), len(bm)); bs.append(1 - zm[ix].mean() / bm[ix].mean())
    res["cut_BAND_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    # head 27.8 answer-row token-0 mass, base vs BAND zeroed in block 27 only
    def ans(**mode):
        V.S.clear(); V.S.update({"mode": "record", "rec": {}}); V.S.update(mode); V.S["rec_too"] = True
        E.run(H); rec = V.S["rec"]; V.S.clear(); V.S["mode"] = "none"; return rec[1][:, 8, -1, :3].mean(0).tolist()
    res["h27_8_ans_t012_base"] = ans()
    res["h27_8_ans_t012_Z27"] = ans(mode="zero", blocks=[1], dims=V.pair_dims(BAND))
    per_head = {}
    for L in (0, 1):
        zL = sink(mode="zero", blocks=[L], dims=V.pair_dims(BAND))
        for h in range(16):
            per_head[f"{26 + L}.{h}"] = [float(base[:, L, h].mean()), float(zL[:, L, h].mean())]
    res["per_head_base_vs_Z"] = per_head
    vn = np.array([per[f"VN{i}"] for i in range(200)]); vu = np.array([per[f"VU{i}"] for i in range(50)])
    res["cut_VN_median_p95"] = [float(np.median(vn)), float(np.percentile(vn, 95))]
    res["cut_VU_median_max"] = [float(np.median(vu)), float(vu.max())]
    out = res
else:
    H = E.CON; don = E.donor()
    P = {"ON": E.px(H), "OFF": E.px(E.COFF)}
    for name, prs in rows.items():
        P[name] = E.px(H, mode="patch", blocks=[0, 1], donor=don, dims=V.pair_dims(prs), pos=[0, 1, 2])
    np.savez(f"{V.HERE}/vera_rows_{seed}_{dt}.npz", **P)
    gap = P["OFF"].mean() - P["ON"].mean()
    sh = {k: float((v.mean() - P["ON"].mean()) / gap) for k, v in P.items() if k not in ("ON", "OFF")}
    rng = np.random.default_rng(0); n = len(P["ON"]); bs = []
    for _ in range(2000):
        ix = rng.integers(0, n, n); bs.append((P["BAND"][ix].mean() - P["ON"][ix].mean()) / (P["OFF"][ix].mean() - P["ON"][ix].mean()))
    vn = np.array([sh[f"VN{i}"] for i in range(200)]); vu = np.array([sh[f"VU{i}"] for i in range(50)])
    S = np.array([sh[f"S{j}"] for j in range(64)])
    out = {"P_on": float(P["ON"].mean()), "P_off": float(P["OFF"].mean()), "gap": float(gap),
           "G": sh["G"], "BAND": sh["BAND"], "BAND_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
           "COMP": sh["COMP"], "LOW8": sh["LOW8"], "SLOW36": sh["SLOW36"],
           "VN_median_p95_max": [float(np.median(vn)), float(np.percentile(vn, 95)), float(vn.max())],
           "VU_median_p95_max": [float(np.median(vu)), float(np.percentile(vu, 95)), float(vu.max())],
           "S": S.tolist(), "S_top5": [[int(j), float(S[j])] for j in np.argsort(-S)[:5]], "S_sum": float(S.sum())}
out["sets"] = {"VN": VN, "VU": VU}
out["sec"] = time.time() - t0
json.dump(out, open(f"{V.HERE}/vera_rows_{seed}_{dt}.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("sets", "S", "cut", "per_head_base_vs_Z")}, indent=1))
