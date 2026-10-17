"""Verdict, baselines, nulls and sensitivity rows from the saved attack runs (PREREG.md + DEVIATIONS.md).

Inputs: data/runs/*__seed*.json (per-run tables written by attack.py), data/clean/, data/probes/, data/fisher/.
Outputs: data/results/pairs.csv (one row per scored pair, seed-mean), data/results/pairs_by_seed.csv,
data/results/results.json (every R^2 and test), figures data via run.py.

Primary set: 5 targets x 49 probes, predictor = exact empirical AUROC map (predict.predicted_drop).
Secondary set: same-concept random-subspace probes, predictor = gap map (predict.predicted_drop_gauss).
R^2 = 1 - SSE/SST around the scored pairs' own mean, no fit. Per pair, predicted and measured drops are the
mean over the target's successful seeds (each seed with its own delta_w).
Run: uv run python projects/fisher-pullback-transfer/analysis.py
"""
import csv
import json

import numpy as np

from common import DATA
from predict import fisher_metric, labels, load_clean, metrics, predicted_drop, predicted_drop_gauss

RES = DATA / "results"
SEED = 0


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def load_runs(sel, tag=""):
    """tag '' = scored runs (pilot lambda); tag 'lam0__' = lambda_KL = 0 diagnostic runs."""
    runs = {}
    for t in sel["targets"]:
        stem = t.replace(":", "__").replace(" ", "_").replace("&", "and")
        runs[t] = []
        for f in sorted((DATA / "runs").glob(f"{stem}__{tag}seed*.json")):
            runs[t].append(json.loads(f.read_text()))
    return runs


def sec_probe(r, W2):
    return {"w": W2[r["idx"]], "source": r["source"], "label": r["label"], "name": r["name"]}


def build_rows(clean, sel, runs, Ps, primary_fn=predicted_drop, secondary_fn=predicted_drop_gauss):
    """Per (set, target, probe, seed): measured drop and predicted drop for each P in Ps (+ shared)."""
    P0 = {p["name"]: p for p in clean["probes"]}
    sec = [r for r in json.loads((DATA / "probes" / "secondary.json").read_text()) if r["kept"]]
    W2 = np.load(DATA / "probes" / "secondary_weights.npz")["W"]
    rows = []
    for t in sel["targets"]:
        w = P0[t]
        for run in runs[t]:
            if not run["success"]:
                continue
            ratio = run["delta_w"] / run["G_w"]
            meas = {(r["set"], r["probe"]): r["auroc_clean"] - r["auroc_attacked"] for r in run["probes"]}
            for name in sel["scored_probes"]:
                if name == t:
                    continue
                row = {"set": "primary", "target": t, "probe": name, "seed": run["seed"], "ratio": ratio,
                       "measured": meas[("primary", name)]}
                for k, P in Ps.items():
                    row[k] = primary_fn(clean, P0[name], w, P, ratio)
                row["shared"] = primary_fn(clean, P0[name], w, None, ratio)
                rows.append(row)
            for r in sec:
                if r["target"] != t:
                    continue
                u = sec_probe(r, W2)
                row = {"set": "secondary", "target": t, "probe": r["name"], "seed": run["seed"], "ratio": ratio,
                       "measured": meas[("secondary", r["name"])]}
                for k, P in Ps.items():
                    row[k] = secondary_fn(clean, u, w, P, ratio)
                row["shared"] = secondary_fn(clean, u, w, None, ratio)
                rows.append(row)
    return rows


def seed_mean(rows, keys):
    out = {}
    for r in rows:
        k = (r["set"], r["target"], r["probe"])
        out.setdefault(k, []).append(r)
    res = []
    for (s, t, p), rs in out.items():
        res.append({"set": s, "target": t, "probe": p, "n_seeds": len(rs),
                    **{k: float(np.mean([r[k] for r in rs])) for k in keys}})
    return res


def verdict(pairs, keys=("plain", "whitened", "fisher", "shared")):
    y = [p["measured"] for p in pairs]
    R = {k: r2(y, [p[k] for p in pairs]) for k in keys}
    m = np.array(y)
    R["sd_measured"] = float(m.std())
    R["n_abs_drop_gt_0.02"] = int((np.abs(m) > 0.02).sum())
    R["n_pairs"] = len(pairs)
    R["margin_vs_best_cosine"] = R["fisher"] - max(R["plain"], R["whitened"])
    R["margin_vs_shared"] = R["fisher"] - R["shared"]
    R["confirmed"] = bool(R["fisher"] >= 0.6 and R["margin_vs_best_cosine"] >= 0.10 and R["margin_vs_shared"] >= 0.10)
    return R


def per_target(pairs, targets, keys=("plain", "whitened", "fisher", "shared")):
    out = {}
    for t in targets:
        ps = [p for p in pairs if p["target"] == t]
        if len(ps) > 2:
            out[t] = {k: r2([p["measured"] for p in ps], [p[k] for p in ps]) for k in keys}
            out[t]["sd_measured"] = float(np.std([p["measured"] for p in ps]))
    return out


def bootstrap(pairs, targets, n=10000, keys=("plain", "whitened", "fisher", "shared")):
    rng = np.random.default_rng(SEED)
    by = {t: [p for p in pairs if p["target"] == t] for t in targets}
    vals = {k: [] for k in list(keys) + ["margin_vs_best_cosine", "margin_vs_shared"]}
    for _ in range(n):
        ps = [p for t in rng.choice(targets, len(targets), replace=True) for p in by[t]]
        if len(ps) < 3 or np.std([p["measured"] for p in ps]) == 0:
            continue
        v = verdict(ps, keys)
        for k in vals:
            vals[k].append(v[k])
    return {k: {"p2.5": float(np.percentile(v, 2.5)), "p50": float(np.percentile(v, 50)),
                "p97.5": float(np.percentile(v, 97.5)), "n": len(v)} for k, v in vals.items()}


def loto_affine(pairs, xkey, targets):
    """Leave-one-target-out affine fit of measured on pairs[xkey]; returns held-out R^2 over all pairs."""
    pred = {}
    if len({p["target"] for p in pairs}) < 2:
        return float("nan")
    for t in targets:
        tr = [p for p in pairs if p["target"] != t]
        a, b = np.polyfit([p[xkey] for p in tr], [p["measured"] for p in tr], 1)
        for p in pairs:
            if p["target"] == t:
                pred[(p["target"], p["probe"])] = a * p[xkey] + b
    return r2([p["measured"] for p in pairs], [pred[(p["target"], p["probe"])] for p in pairs])


def direction_predictors(clean, sel, runs):
    """EXPLORATORY: the same formula with P w replaced by a direction v (rank-1 P = v v', so r = u'v / w'v).
    v = K w (tangent kernel, plain gradient step), K_adam w (Adam's first step), both at the clean weights
    (kernel_dir.py), and v = the run's own measured class-mean edit (an oracle: it uses the attack's outcome)."""
    P0 = {p["name"]: p for p in clean["probes"]}
    sec = [r for r in json.loads((DATA / "probes" / "secondary.json").read_text()) if r["kept"]]
    W2 = np.load(DATA / "probes" / "secondary_weights.npz")["W"]
    out = {}
    for tag in ("Kw", "Kadam", "measured_edit"):
        rows = []
        for t in sel["targets"]:
            stem = t.replace(":", "__").replace(" ", "_").replace("&", "and")
            for run in runs[t]:
                if not run["success"]:
                    continue
                if tag == "measured_edit":
                    e = np.load(DATA / "runs" / f"{stem}__seed{run['seed']}.npz")
                    v = e["mean_edit_pos"] - e["mean_edit_neg"]
                else:
                    v = np.load(DATA / "runs" / f"{tag}__{stem}.npy")
                P = np.outer(v, v)
                ratio = run["delta_w"] / run["G_w"]
                meas = {(r["set"], r["probe"]): r["auroc_clean"] - r["auroc_attacked"] for r in run["probes"]}
                for name in sel["scored_probes"]:
                    if name != t:
                        rows.append({"set": "primary", "target": t, "probe": name, "seed": run["seed"],
                                     "measured": meas[("primary", name)],
                                     "pred": predicted_drop(clean, P0[name], P0[t], P, ratio)})
                for r in sec:
                    if r["target"] == t:
                        rows.append({"set": "secondary", "target": t, "probe": r["name"], "seed": run["seed"],
                                     "measured": meas[("secondary", r["name"])],
                                     "pred": predicted_drop_gauss(clean, sec_probe(r, W2), P0[t], P, ratio)})
        pm = seed_mean(rows, ["measured", "pred"])
        out[tag] = {s: r2([p["measured"] for p in pm if p["set"] == s], [p["pred"] for p in pm if p["set"] == s])
                    for s in ("primary", "secondary")}
    return out


def main():
    RES.mkdir(parents=True, exist_ok=True)
    sel = json.loads((DATA / "selection.json").read_text())
    clean = load_clean()
    runs = load_runs(sel)
    targets = sel["targets"]
    Ps = metrics()
    keys = ["measured", "plain", "whitened", "fisher", "shared", "ratio"]
    rows = build_rows(clean, sel, runs, Ps)
    pairs = seed_mean(rows, keys)
    prim = [p for p in pairs if p["set"] == "primary"]
    secd = [p for p in pairs if p["set"] == "secondary"]

    run_summary = [{k: r.get(k) for k in ("target", "seed", "success", "stop_step", "lambda_kl", "target_auroc_clean",
                                          "target_auroc_attacked", "delta_w", "G_w", "retain_kl_heldout", "device")}
                   for t in targets for r in runs[t]]
    n_ok = {t: sum(r["success"] for r in runs[t]) for t in targets}
    status_ok = all(v >= 2 for v in n_ok.values())
    res = {"runs": run_summary, "successful_seeds": n_ok}
    res["primary"] = verdict(prim)
    res["primary"]["status"] = ("CONFIRMED" if res["primary"]["confirmed"] else "NULL") if status_ok else "UNCONFIRMED"
    res["primary"]["per_target"] = per_target(prim, targets)
    res["primary"]["bootstrap_targets"] = bootstrap(prim, targets)
    if secd:
        res["secondary"] = verdict(secd)
        res["secondary"]["per_target"] = per_target(secd, targets)
        res["secondary"]["bootstrap_targets"] = bootstrap(secd, [t for t in targets if any(p["target"] == t for p in secd)])

    # --- sensitivity rows (not the verdict) ---
    sens = {}
    rows_g = build_rows(clean, sel, runs, Ps, primary_fn=predicted_drop_gauss, secondary_fn=predicted_drop)
    pg = seed_mean(rows_g, keys)
    sens["primary_gauss_map"] = verdict([p for p in pg if p["set"] == "primary"])
    sens["secondary_empirical_map"] = verdict([p for p in pg if p["set"] == "secondary"]) if secd else None
    damp = {}
    for c in (0.01, 0.1, 1.0, 10.0, 100.0):
        rr = seed_mean(build_rows(clean, sel, runs, {"fisher": fisher_metric(c)}), ["measured", "fisher", "shared"])
        damp[f"c={c}"] = {s: r2([p["measured"] for p in rr if p["set"] == s], [p["fisher"] for p in rr if p["set"] == s])
                          for s in ("primary", "secondary") if any(p["set"] == s for p in rr)}
    z = np.load(DATA / "fisher" / "fisher_sigma.npz")
    zb = np.load(DATA / "fisher" / "fisher_big.npz")
    Fc = zb["F_common"]
    med = lambda M: np.linalg.inv(M + np.median(np.linalg.eigvalsh(M)) * np.eye(512))
    for tag, M in (("lambda=1e-2*tr/512", np.linalg.inv(Fc + 1e-2 * np.trace(Fc) / 512 * np.eye(512))),
                   ("F_token c=1", fisher_metric(1.0, "F_token")),
                   ("F_common half1 c=1", med(zb["F_common_half1"])),
                   ("F_common half2 c=1", med(zb["F_common_half2"])),
                   ("F_common 2000 windows c=1", fisher_metric(1.0, "F_common", "fisher_sigma.npz"))):
        rr = seed_mean(build_rows(clean, sel, runs, {"fisher": M}), ["measured", "fisher"])
        damp[tag] = {s: r2([p["measured"] for p in rr if p["set"] == s], [p["fisher"] for p in rr if p["set"] == s])
                     for s in ("primary", "secondary") if any(p["set"] == s for p in rr)}
    sens["fisher_variants_R2"] = damp
    res["sensitivity"] = sens

    # --- baselines (not the verdict, except shared above) ---
    P0 = {p["name"]: p for p in clean["probes"]}
    base = {}
    for p in prim:
        u, w = P0[p["probe"]], P0[p["target"]]
        for k, P in Ps.items():
            p[f"cos_{k}"] = float(u["w"] @ P @ w["w"] / np.sqrt((u["w"] @ P @ u["w"]) * (w["w"] @ P @ w["w"])))
        X = clean["acts"][u["source"]]["X"]
        p["labelcorr"] = abs(float(np.corrcoef(labels(clean, u), X @ w["w"])[0, 1]))
        others = [q["measured"] for q in prim if q["probe"] == p["probe"] and q["target"] != p["target"]]
        p["fragility"] = float(np.mean(others)) if others else 0.0
    for k in Ps:
        base[f"literal_cos_{k}_affine_LOTO"] = loto_affine(prim, f"cos_{k}", targets)
        base[f"literal_abscos_{k}_affine_LOTO"] = loto_affine([dict(p, **{f"a_{k}": abs(p[f"cos_{k}"])}) for p in prim], f"a_{k}", targets)
    base["labelcorr_affine_LOTO"] = loto_affine(prim, "labelcorr", targets)
    base["fragility_noFit"] = r2([p["measured"] for p in prim], [p["fragility"] for p in prim])
    res["baselines_primary"] = base

    # --- nulls ---
    rng = np.random.default_rng(SEED)
    Fd = fisher_metric(1.0)
    ev = np.linalg.eigvalsh(Fd)
    rnd = []
    for _ in range(100):
        Q = np.linalg.qr(rng.standard_normal((512, 512)))[0]
        rr = seed_mean(build_rows(clean, sel, runs, {"fisher": Q @ np.diag(ev) @ Q.T}), ["measured", "fisher"])
        rnd.append({s: r2([p["measured"] for p in rr if p["set"] == s], [p["fisher"] for p in rr if p["set"] == s])
                    for s in ("primary", "secondary") if any(p["set"] == s for p in rr)})
    nulls = {"random_rotation_metric": {s: {"mean": float(np.mean([d[s] for d in rnd])),
                                            "p95": float(np.percentile([d[s] for d in rnd], 95)),
                                            "frac_ge_actual": float(np.mean([d[s] >= res[s]["fisher"] for d in rnd]))}
                                        for s in rnd[0]}}
    perm = {}
    for s, ps in (("primary", prim), ("secondary", secd)):
        if not ps:
            continue
        y = np.array([p["measured"] for p in ps])
        f = np.array([p["fisher"] for p in ps])
        tg = np.array([p["target"] for p in ps])
        vals = []
        for _ in range(1000):
            g = f.copy()
            for t in set(tg):
                i = np.where(tg == t)[0]
                g[i] = f[rng.permutation(i)]
            vals.append(r2(y, g))
        perm[s] = {"mean": float(np.mean(vals)), "p95": float(np.percentile(vals, 95)),
                   "frac_ge_actual": float(np.mean(np.array(vals) >= res[s]["fisher"]))}
    nulls["within_target_permutation"] = perm
    res["nulls"] = nulls

    # --- exploratory: measured edit direction ---
    Sig = z["Sigma_LW"]
    ones = np.ones(512) / np.sqrt(512)
    expl = []
    for t in targets:
        w = P0[t]["w"]
        dirs = {"w": w, "Sigma_w": Sig @ w, "Finv_w": Fd @ w, "ones": ones}
        for r in runs[t]:
            stem = t.replace(":", "__").replace(" ", "_").replace("&", "and")
            e = np.load(DATA / "runs" / f"{stem}__seed{r['seed']}.npz")
            d = e["mean_edit_pos"] - e["mean_edit_neg"]
            a = e["mean_edit_all"]
            row = {"target": t, "seed": r["seed"], "success": r["success"],
                   "edit_norm": float(np.linalg.norm(d)), "mean_edit_all_norm": float(np.linalg.norm(a)),
                   "mean_edit_all_cos_ones": float(a @ ones / np.linalg.norm(a))}
            for k, v in dirs.items():
                row[f"cos_classedit_{k}"] = float(d @ v / np.linalg.norm(d) / np.linalg.norm(v))
            for tag in ("Kw", "Kadam"):
                kw = DATA / "runs" / f"{tag}__{stem}.npy"
                if kw.exists():
                    v = np.load(kw)
                    row[f"cos_classedit_{tag}"] = float(d @ v / np.linalg.norm(d) / np.linalg.norm(v))
            expl.append(row)
    res["exploratory_edit_direction"] = expl
    res["exploratory_direction_predictors"] = direction_predictors(clean, sel, runs)
    res["probe_ones_cos"] = {p["name"]: p.get("ones_cos") for p in clean["probes"]}

    # --- lambda_KL = 0 diagnostic (DEVIATIONS 15:13): retain-KL ratio and the predictors on these runs ---
    runs0 = load_runs(sel, "lam0__")
    kl = []
    for t in targets:
        a = [r for r in runs[t] if r["seed"] == 0]
        b = [r for r in runs0[t] if r["seed"] == 0]
        if a and b:
            kl.append({"target": t, "kl_chosen": a[0]["retain_kl_heldout"], "kl_lambda0": b[0]["retain_kl_heldout"],
                       "ratio": a[0]["retain_kl_heldout"] / b[0]["retain_kl_heldout"],
                       "stop_step_chosen": a[0]["stop_step"], "stop_step_lambda0": b[0]["stop_step"],
                       "success_lambda0": b[0]["success"]})
    lam0 = {"kl_ratio_per_target": kl}
    if any(runs0.values()):
        p0 = seed_mean(build_rows(clean, sel, runs0, Ps), keys)
        lam0["primary"] = verdict([p for p in p0 if p["set"] == "primary"])
        lam0["secondary"] = verdict([p for p in p0 if p["set"] == "secondary"])
    res["lambda0_diagnostic"] = lam0

    (RES / "results.json").write_text(json.dumps(res, indent=1))
    for name, data in (("pairs.csv", pairs), ("pairs_by_seed.csv", rows)):
        with open(RES / name, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(data[0].keys()) if data else ["empty"], extrasaction="ignore")
            wr.writeheader()
            wr.writerows(data)
    print(json.dumps({k: res[k] for k in ("successful_seeds",)}, indent=1))
    for s in ("primary", "secondary"):
        if s in res:
            print(s, json.dumps({k: v for k, v in res[s].items() if k not in ("per_target", "bootstrap_targets")}))
    print("baselines", json.dumps(base))
    print("nulls", json.dumps(nulls))
    print("fisher variants", json.dumps(damp))


if __name__ == "__main__":
    main()
