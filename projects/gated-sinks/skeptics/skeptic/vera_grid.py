"""Vera: grid-level rebuild. Reads my per-model rebuild json (vera_rebuild.py output) + build's measure jsons
and results/lstar.json; computes: matched-loss check, PREREG s3 law/status (build SAEs and my own SAEs),
registered-prediction outcome (s1), leave-one-out stability of the status.
"""
import argparse, json, os
import numpy as np
from vera_status import law_status

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--rebuild", nargs="+", required=True)
p.add_argument("--out", default="vera_grid.json")
p.add_argument("--x", default="sinkblk", choices=["sinkblk", "blk2", "allmean"])
a = p.parse_args()

rows = []
for f in a.rebuild:
    rows += json.load(open(f))
lstar_path = os.path.join(a.root, "results", "lstar.json")
lstar = json.load(open(lstar_path)) if os.path.exists(lstar_path) else None
L = None
if lstar:
    L = lstar.get("L_star") or lstar.get("lstar") or lstar.get("target")
out = dict(L_star=L, models=[])
for r in rows:
    m = json.load(open(os.path.join(a.root, r["npz"].replace("acts_", "measure_").replace(".npz", ".json"))))
    gap = None if L is None else (L - m["eval_loss"]) / L
    out["models"].append(dict(run=r["run"], arm=r["arm"], seed=r["seed"], gated=r["arm"] == "gated",
                              sink=(r["D_sink"] if a.x == "sinkblk" else r["D_sink_blocks"][1] if a.x == "blk2" else r["D_sink_allblock_mean"]), rec_build=r["A_build_sae_subbdec"]["recovery"],
                              rec_own=r["B_own_sae"]["recovery"], hung=r["A_build_sae_subbdec"]["hungarian"],
                              floor=r["C_random_dict_floor"], shuffled=r.get("C_shuffled_label_build_sae0"),
                              eval_loss=m["eval_loss"], loss_gap=gap, json_rec=m["recovery"], json_sink=m["sink_mass"],
                              norm_r2=m.get("norm_r2"), ckpt=m.get("ckpt"), read_corr=r["E_readout"]["corr_mean"],
                              gap=r["E_readout"]["gap"], eff=r["E_readout"]["efficiency"], json_gap=m.get("readability_gap")))
ms = out["models"]
for key in ("rec_build", "rec_own", "hung"):
    st = law_status([dict(run=x["run"], arm=x["arm"], gated=x["gated"], sink=x["sink"], recovery=x[key]) for x in ms])
    out[f"status_{key}"] = st
    # leave-one-out over non-gated models
    loo = []
    for i, x in enumerate(ms):
        if x["gated"]:
            continue
        sub = [dict(run=y["run"], arm=y["arm"], gated=y["gated"], sink=y["sink"], recovery=y[key]) for j, y in enumerate(ms) if j != i]
        loo.append((x["run"], law_status(sub)["status"]))
    out[f"loo_{key}"] = loo
# registered prediction (s1): pairs by seed
sm = {x["seed"]: x for x in ms if x["arm"] == "softmax"}
gt = {x["seed"]: x for x in ms if x["arm"] == "gated"}
pairs = []
for s in sorted(set(sm) & set(gt)):
    pairs.append(dict(seed=s, d_rec=gt[s]["rec_build"] - sm[s]["rec_build"], d_rec_own=gt[s]["rec_own"] - sm[s]["rec_own"],
                      sink_ratio=gt[s]["sink"] / sm[s]["sink"] if sm[s]["sink"] > 0 else None))
out["prereg_pairs"] = pairs
out["prereg_a"] = all(p_["d_rec"] >= 0.10 for p_ in pairs) and len(pairs) == 3
out["prereg_b"] = all(p_["sink_ratio"] is not None and p_["sink_ratio"] < 0.2 for p_ in pairs) and len(pairs) == 3
out["prereg_c"] = all((x["norm_r2"] or 0) < 0.2 for x in ms if x["arm"] in ("softmax", "gated"))
out["prereg_outcome"] = "held" if (out["prereg_a"] and out["prereg_b"] and out["prereg_c"]) else "failed"
# named row: same architecture, different sink (softmax seeds only)
from scipy import stats as _st
sx = [x for x in ms if x["arm"] == "softmax"]
if len(sx) >= 3:
    xs_ = np.array([x["sink"] for x in sx]); 
    for key in ("rec_build", "rec_own"):
        ys_ = np.array([x[key] for x in sx])
        lr = _st.linregress(xs_, ys_)
        out[f"within_softmax_{key}"] = dict(n=len(sx), slope=lr.slope, slope_se=lr.stderr, p=lr.pvalue,
                                            spearman=_st.spearmanr(xs_, ys_).correlation)
        print("within-softmax", key, out[f"within_softmax_{key}"])
# secondary (Felix #17311): readability gap ~ sink and efficiency ~ sink, non-gated
for key in ("gap", "eff", "read_corr"):
    ngx = [x for x in ms if not x["gated"]]
    if len(ngx) >= 3:
        lr = _st.linregress([x["sink"] for x in ngx], [x[key] for x in ngx])
        t7 = _st.t.ppf(0.975, len(ngx) - 2)
        out[f"secondary_{key}_on_sink"] = dict(slope=lr.slope, ci=[lr.slope - t7 * lr.stderr, lr.slope + t7 * lr.stderr], n=len(ngx))
        print("secondary", key, out[f"secondary_{key}_on_sink"])
# null: permutation test of the non-gated slope (sink labels shuffled across models), 20k perms
_rng = np.random.default_rng(0)
for key in ("rec_build", "rec_own"):
    ngp = [x for x in ms if not x["gated"]]
    if len(ngp) >= 4:
        xp = np.array([x["sink"] for x in ngp]); yp = np.array([x[key] for x in ngp])
        b_obs = np.polyfit(xp, yp, 1)[0]
        bs = np.array([np.polyfit(_rng.permutation(xp), yp, 1)[0] for _ in range(20000)])
        out[f"perm_{key}"] = dict(b_obs=b_obs, p_two_sided=float((np.abs(bs) >= abs(b_obs) - 1e-12).mean()), n=len(ngp))
        print("perm", key, out[f"perm_{key}"])
        # binary-label null: which non-gated models carry a sink (x > 0.3); exact rank p of their mean recovery
        hi = xp > 0.3
        if 0 < hi.sum() < len(xp):
            from itertools import combinations
            obs = yp[hi].mean()
            vals = [yp[list(c)].mean() for c in combinations(range(len(yp)), int(hi.sum()))]
            out[f"perm_binary_{key}"] = dict(n_sink=int(hi.sum()), obs_mean=obs, p_one_sided=float(np.mean([v >= obs - 1e-12 for v in vals])))
            print("perm_binary", key, out[f"perm_binary_{key}"])
# pre-declared sensitivity (#17298): recovery ~ sink + loss on non-gated models
ng_ = [x for x in ms if not x["gated"]]
if len(ng_) >= 5:
    X_ = np.column_stack([np.ones(len(ng_)), [x["sink"] for x in ng_], [x["eval_loss"] for x in ng_]])
    for key in ("rec_build", "rec_own"):
        y_ = np.array([x[key] for x in ng_])
        beta, *_ = np.linalg.lstsq(X_, y_, rcond=None)
        res_ = y_ - X_ @ beta
        dof = len(y_) - 3
        cov = (res_ @ res_ / dof) * np.linalg.inv(X_.T @ X_)
        t_ = _st.t.ppf(0.975, dof)
        se = np.sqrt(np.diag(cov))
        out[f"sink_plus_loss_{key}"] = dict(b_sink=beta[1], ci_sink=[beta[1] - t_ * se[1], beta[1] + t_ * se[1]],
                                            b_loss=beta[2], ci_loss=[beta[2] - t_ * se[2], beta[2] + t_ * se[2]], dof=dof)
        print("sink+loss", key, out[f"sink_plus_loss_{key}"])
json.dump(out, open(a.out, "w"), indent=1, default=float)
for x in ms:
    print(f"{x['run']:28s} {x['arm']:9s} s{x['seed']} sink {x['sink']:.3f} rec {x['rec_build']:.4f} (json {x['json_rec']:.4f}) own {x['rec_own']:.4f} rc {x['read_corr']:.3f} gap {x['gap']:.3f} eff {x['eff']:.3f} floor {x['floor']:.3f} shuf {x['shuffled'] if x['shuffled'] is None else round(x['shuffled'],3)} loss {x['eval_loss']:.4f} gap {x['loss_gap'] if x['loss_gap'] is None else round(x['loss_gap']*100,2)}%")
for key in ("rec_build", "rec_own", "hung"):
    st = out[f"status_{key}"]
    print(key, st["status"], "b", round(st["b"], 4), "CI", [round(v, 4) for v in st["b_ci"]], "s_tot", round(st["s_tot"], 4), "R", round(st["R"], 3),
          "swing_hi", round(st["swing_hi"], 4), "gated in band", st["n_in"], "levels", st["sink_levels"],
          "resid", [round(g["resid"], 4) for g in st["gated"]], "band", [round(g["band"], 4) for g in st["gated"]], "LOO", out[f"loo_{key}"])
print("prereg", out["prereg_outcome"], out["prereg_pairs"])
