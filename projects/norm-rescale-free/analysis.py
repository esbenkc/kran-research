"""Every reported number from the saved run files (data/runs/<name>/ckpt_<step>.npz + acts_<step>.npy).
All probe scores are recomputed here in float64 from the saved fp16 country-token states (bf16 -> fp16 is exact), so a
skeptic needs only the states, data/fits/panel.npz and data/ref/split.json.

Title metric (PREREG, Felix #5435 / Hugo #5430), per run, per checkpoint, readout 28, frozen probe tok28:
    s(h)       = ((h - mu) / sd) . w + b
    g(state)   = mean over the 596 eval twin pairs (same city, same template) of s(false) - s(true twin)
    restored   = every attacked state h' rescaled to its clean length: h'' = h' |h| / |h'|  (true and false)
    R          = (g_restored - g_attacked) / (g_clean - g_attacked)
    precondition: gap_drop_frac = (g_clean - g_attacked) / g_clean >= 0.5
Catch: eval false statements flagged iff s > line; line = 2nd-highest of the 200 line-set true statements, either of
the scored state (catch_reset) or of the clean state (catch_clean).
Eval KL: mean over every position of the 1192 eval statements of KL(p_clean || p_state) (in-run, batch-matched).
Energy split (support rows, not the title): for each eval false statement, d = h' - h at readout 28,
radial = (d . h/|h|)^2 / |d|^2; pooled = sum radial energy / sum |d|^2.
Output: data/results/results.json, data/results/per_statement_<name>.npz
"""
import json
from pathlib import Path

import numpy as np

from common import DATA, RUNS, rows, threshold
from metrics import boot_catch, ranking

OUT = DATA / "results"
OUT.mkdir(parents=True, exist_ok=True)
F = np.load(DATA / "fits" / "panel.npz")
PROBE = {R: (F[f"tok{R}__w"].astype(np.float64), float(F[f"tok{R}__b"]), F[f"tok{R}__mu"].astype(np.float64),
             F[f"tok{R}__sd"].astype(np.float64)) for R in (27, 28, 29)}
R_, y, city, split = rows()
src = np.array([r["src"] for r in R_])
line_i = np.where((split == "refit") & (y == 0))[0]
ev = np.where(split == "eval")[0]
ev_f = np.where((split == "eval") & (y == 1))[0]
ev_t = np.where((split == "eval") & (y == 0))[0]
LAST = {}  # run -> {28: tok28 scores, 29: tok29 scores} at its last checkpoint (all 2992 statements)
pairs = []
for i in ev_f:
    j = np.where((split == "eval") & (y == 0) & (city == city[i]) & (src == src[i]))[0]
    assert len(j) == 1
    pairs.append((i, int(j[0])))
pairs = np.array(pairs)


def score(R, X):
    w, b, mu, sd = PROBE[R]
    return ((X - mu) / sd) @ w + b


def acts(run, step, R):
    ck = np.load(RUNS / run / f"ckpt_{step}.npz")
    A = np.load(RUNS / run / f"acts_{step}.npy", mmap_mode="r")
    j = list(ck["acts_readouts"]).index(R)
    return np.asarray(A[:, j], np.float64), ck


def gap(s):
    return float(np.mean(s[pairs[:, 0]] - s[pairs[:, 1]]))


def radial(X0, X1, idx):
    D = X1[idx] - X0[idx]
    u = X0[idx] / np.linalg.norm(X0[idx], axis=1, keepdims=True)
    r = (D * u).sum(1)
    n2 = (D ** 2).sum(1)
    return r, n2


def analyse(run):
    log = json.loads((RUNS / run / "log.json").read_text())
    steps = sorted(int(p.stem.split("_")[1]) for p in (RUNS / run).glob("ckpt_*.npz"))
    X0, ck0 = acts(run, 0, 28)
    s0 = score(28, X0)
    s0_29 = score(29, acts(run, 0, 29)[0])
    g_clean = gap(s0)
    line_clean = threshold(s0[line_i])
    line_clean_29 = threshold(s0_29[line_i])
    koff, cpos = ck0["koff"], ck0["cpos"]
    rows_out = []
    per = {}
    for st in steps:
        X1, ck = acts(run, st, 28)
        s1 = score(28, X1)
        n0, n1 = np.linalg.norm(X0, axis=1), np.linalg.norm(X1, axis=1)
        s_rest = score(28, X1 * (n0 / n1)[:, None])
        g_att, g_rest = gap(s1), gap(s_rest)
        den = g_clean - g_att
        s1_29 = score(29, acts(run, st, 29)[0])
        kl = ck["kl"]
        evkl = np.concatenate([kl[koff[i]:koff[i + 1]] for i in ev])
        r, n2 = radial(X0, X1, ev_f)
        frac = np.divide(r ** 2, n2, out=np.zeros_like(r), where=n2 > 0)
        cos = np.divide(r, np.sqrt(n2), out=np.zeros_like(r), where=n2 > 0)
        lc_f, lc_t = np.log(n1[pairs[:, 0]] / n0[pairs[:, 0]]), np.log(n1[pairs[:, 1]] / n0[pairs[:, 1]])
        # in-run (MPS, bf16 states, fp32 math) tok28 scores vs the float64 rebuild here
        inrun_diff = float(np.abs(ck["scores"][:, list(ck["scored"]).index(28)] - s1).max())
        # PREREG s8: one common rescale factor (mean clean/attacked norm over the 200 line-setting true statements)
        cf = float(np.mean(n0[line_i] / n1[line_i]))
        g_com = gap(score(28, X1 * cf))
        # PREREG s8: exact radial share of the score change on eval false: h' = c h + t, t _|_ h
        wsd = PROBE[28][0] / PROBE[28][3]
        a_i = X0[ev_f] @ wsd
        c_i = (X0[ev_f] * X1[ev_f]).sum(1) / n0[ev_f] ** 2
        ds = s1[ev_f] - s0[ev_f]
        line_r = threshold(s1[line_i])
        line_r29 = threshold(s1_29[line_i])
        rows_out.append({
            "step": st, "R": (g_rest - g_att) / den if abs(den) > 1e-9 else None,
            "g_clean": g_clean, "g_attacked": g_att, "g_restored": g_rest, "gap_drop_frac": den / g_clean,
            "catch_reset": float((s1[ev_f] > threshold(s1[line_i])).mean()),
            "catch_clean": float((s1[ev_f] > line_clean).mean()),
            "catch_reset_tok29": float((s1_29[ev_f] > threshold(s1_29[line_i])).mean()),
            "catch_clean_tok29": float((s1_29[ev_f] > line_clean_29).mean()),
            "catch_reset_restored": float((s_rest[ev_f] > threshold(s_rest[line_i])).mean()),
            "false_alarm_reset": float((s1[ev_t] > line_r).mean()),
            "false_alarm_clean": float((s1[ev_t] > line_clean).mean()),
            "false_alarm_reset_tok29": float((s1_29[ev_t] > line_r29).mean()),
            "R_common": (g_com - g_att) / den if abs(den) > 1e-9 else None, "common_factor": cf,
            "score_change_radial_share": float(((c_i - 1) * a_i).sum() / ds.sum()) if abs(ds.sum()) > 1e-9 else None,
            "score_change_false_mean": float(ds.mean()),
            "eval_kl": float(evkl.mean()),
            "eval_kl_country": float(np.mean([kl[koff[i] + cpos[i]] for i in ev])),
            "eval_kl_other_positions": float(np.mean(np.concatenate(
                [np.delete(kl[koff[i]:koff[i + 1]], cpos[i]) for i in ev]))),
            "radial_energy_frac": float((r ** 2).sum() / n2.sum()) if n2.sum() > 0 else 0.0,
            "radial_num": float((r ** 2).sum()), "radial_den": float(n2.sum()),
            "radial_energy_frac_median": float(np.median(frac)),
            "radial_cos_median": float(np.median(cos)),
            "norm_ratio_false": float(np.median(n1[ev_f] / n0[ev_f])),
            "norm_ratio_true": float(np.median(n1[pairs[:, 1]] / n0[pairs[:, 1]])),
            "log_norm_ratio_false_minus_true": float(np.mean(lc_f - lc_t)),
            "rel_change_median": float(np.median(np.sqrt(n2) / n0[ev_f])),
            "inrun_score_maxdiff": inrun_diff,
        })
        if st == steps[-1]:
            per = dict(idx=ev_f, radial_frac=frac, radial_cos=cos, norm_ratio=n1[ev_f] / n0[ev_f],
                       s_clean=s0[ev_f], s_attacked=s1[ev_f], s_restored=s_rest[ev_f])
            LAST[run] = {28: s1, 29: s1_29, "line_clean": {28: line_clean, 29: line_clean_29}}
            if "lp" in ck.files:
                rk = ranking(ck["lp"], koff)
                rows_out[-1]["right_first"] = rk["right_first"]
                rows_out[-1]["rank_margin_median"] = rk["margin_median"]
            # PREREG s8 line-resample null: 2000 resamples of the 200 line-setting true statements, seed 1023
            rng = np.random.default_rng(1023)
            res = []
            for _ in range(2000):
                t = threshold(s1[rng.choice(line_i, len(line_i), replace=True)])
                res.append((s1[ev_f] > t).mean())
            rows_out[-1]["catch_reset_line_resample_95"] = [float(np.quantile(res, .025)), float(np.quantile(res, .975))]
    # radial energy share by layer at the last full checkpoint
    by_layer = {}
    last_full = max(s for s in steps if s in (0, 150, 300))
    if last_full > 0:
        A0 = np.load(RUNS / run / "acts_0.npy", mmap_mode="r")
        A1 = np.load(RUNS / run / f"acts_{last_full}.npy", mmap_mode="r")
        for R in range(1, 30):
            r, n2 = radial(np.asarray(A0[:, R], np.float64), np.asarray(A1[:, R], np.float64), ev_f)
            by_layer[R] = float((r ** 2).sum() / n2.sum()) if n2.sum() > 0 else 0.0
    np.savez(OUT / f"per_statement_{run}.npz", **per)
    return {"arm": "pre" if log["readout"] == 28 else "post", "readout": log["readout"], "lam": log["lam"],
            "seed": log["seed"], "kl_pile_heldout": log.get("kl_pile_heldout_mean"), "checkpoints": rows_out,
            "radial_by_layer_step": last_full, "radial_by_layer": by_layer,
            "train_seconds": log["train_seconds"], "max_rss_mb": log["max_rss_mb"]}


def nulls(run):
    """Radial energy share of reference directions at readout 28, eval false statements, clean states."""
    X0, _ = acts(run, 0, 28)
    u = X0[ev_f] / np.linalg.norm(X0[ev_f], axis=1, keepdims=True)
    rng = np.random.default_rng(1023)
    G = rng.standard_normal(u.shape)
    w = PROBE[28][0] / PROBE[28][3]
    D_tf = X0[pairs[:, 0]] - X0[pairs[:, 1]]
    U = rng.standard_normal((1000, u.shape[1]))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    uu = u[np.arange(1000) % len(u)]
    return {"random_gaussian": float(((G * u).sum(1) ** 2).sum() / (G ** 2).sum()),
            "random_unit_1000_seed1023": float(((U * uu).sum(1) ** 2).sum() / 1000),
            "one_over_d": 1 / X0.shape[1],
            "probe_gradient_w_over_sd": float(np.median((u @ w) ** 2 / (w @ w))),
            "probe_gradient_w_over_sd_pooled": float(((u @ w) ** 2).sum() / (len(u) * (w @ w))),
            "false_minus_true_twin": float(((D_tf * u).sum(1) ** 2).sum() / (D_tf ** 2).sum())}


def boot_clean_line(scores, lines, n=2000, seed=1023):
    """Pooled catch at fixed (clean) lines, city-cluster bootstrap as metrics.boot_catch."""
    from metrics import EVPOS, EVPOS_C, EV_CITIES
    hits = np.stack([(s[EVPOS] > t).astype(float) for s, t in zip(scores, lines)])
    nc = len(EV_CITIES)
    ph = sum(np.bincount(EVPOS_C, weights=h, minlength=nc) for h in hits)
    pn = np.bincount(EVPOS_C, minlength=nc).astype(float)
    d = np.random.default_rng(seed).integers(0, nc, size=(n, nc))
    bs = ph[d].sum(1) / (pn[d].sum(1) * len(scores))
    return {"catch": float(hits.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
            "per_seed": [float(h.mean()) for h in hits]}


def pooled(res, R, lam="10000"):
    names = [f"r{R}__lam{lam}__seed{s}" for s in range(3) if f"r{R}__lam{lam}__seed{s}" in LAST]
    out = {"runs": names}
    if not names:
        return out
    for P in (28, 29):
        out[f"catch_reset_tok{P}"] = boot_catch([LAST[n][P] for n in names], seed=1023)
        out[f"catch_clean_tok{P}"] = boot_clean_line([LAST[n][P] for n in names],
                                                     [LAST[n]["line_clean"][P] for n in names])
    cks = [res["runs"][n]["checkpoints"][-1] for n in names]
    out["eval_kl_mean"] = float(np.mean([c["eval_kl"] for c in cks]))
    out["radial_energy_pooled"] = float(sum(c["radial_num"] for c in cks) / sum(c["radial_den"] for c in cks))
    # clause 4: per seed K = smallest eval KL among checkpoints 50..300 with headline catch (own probe) < 20%
    key = "catch_reset" if R == 28 else "catch_reset_tok29"
    K = []
    for n in names:
        ok = [c["eval_kl"] for c in res["runs"][n]["checkpoints"] if c["step"] >= 50 and c[key] < 0.20]
        K.append(min(ok) if ok else None)
    out["K_per_seed"] = K
    return out


def main():
    runs = sorted(p.name for p in RUNS.iterdir() if (p / "log.json").exists())
    res = {"runs": {r: analyse(r) for r in runs}, "n_eval_pairs": int(len(pairs))}
    if runs:
        res["nulls_readout28_clean"] = nulls(runs[0])
    title = {}
    for arm, R in (("A", 28), ("B", 29)):
        rr = [res["runs"][f"r{R}__lam10000__seed{s}"]["checkpoints"][-1] for s in range(3)
              if f"r{R}__lam10000__seed{s}" in res["runs"]]
        title[arm] = {"R_last": [c["R"] for c in rr], "step": [c["step"] for c in rr],
                      "precondition": [c["gap_drop_frac"] >= 0.5 for c in rr]}
    a = title["A"]
    if len(a["R_last"]) == 3 and all(s == 300 for s in a["step"]):
        if not all(a["precondition"]):
            v = "UNCONFIRMED (attack failed precondition)"
        elif all(x >= 0.5 for x in a["R_last"]):
            v = "YES"
        elif all(x < 0.5 for x in a["R_last"]):
            v = "NO"
        else:
            v = "UNCONFIRMED (split)"
        title["verdict"] = v
    res["title"] = title
    res["pooled"] = {"A": pooled(res, 28), "B": pooled(res, 29)}
    pA, pB = res["pooled"]["A"], res["pooled"]["B"]
    if len(pA["runs"]) == 3 and len(pB["runs"]) == 3:
        KA = [k for k in pA["K_per_seed"] if k is not None]
        KB = [k for k in pB["K_per_seed"] if k is not None]
        if len(KA) < 2:
            c4 = False
        elif not KB:
            c4 = True
        else:
            c4 = bool(np.median(KB) / np.median(KA) >= 2)
        cl = {"1_catch_lt_20_and_kl_le_0.002": bool(pA["catch_reset_tok28"]["catch"] < 0.20 and pA["eval_kl_mean"] <= 0.002),
              "2_energy_gt_0.70": bool(pA["radial_energy_pooled"] > 0.70),
              "3_random_lt_0.05": bool(res["nulls_readout28_clean"]["random_unit_1000_seed1023"] < 0.05),
              "4_post_needs_2x_kl": c4,
              "K_ratio": (float(np.median(KB) / np.median(KA)) if KA and KB else None)}
        cl["outcome"] = "held" if all(v for k, v in cl.items() if k[0].isdigit()) else "failed"
        res["prereg_clauses"] = cl
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    for r, v in res["runs"].items():
        c = v["checkpoints"][-1]
        print(r, {k: (round(c[k], 4) if isinstance(c[k], float) else c[k]) for k in
                  ("step", "R", "gap_drop_frac", "catch_reset", "catch_clean", "catch_reset_tok29", "eval_kl",
                   "eval_kl_country", "radial_energy_frac", "norm_ratio_false", "norm_ratio_true")})
    print(json.dumps(res.get("nulls_readout28_clean")), json.dumps(title))


if __name__ == "__main__":
    main()
