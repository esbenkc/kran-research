"""Build results/results.json from the measured runs: matched-loss check, law fit, bands, status.

uv run python aggregate.py [--tag L2_m256_k8]

Implements PREREG.md section 3 exactly: OLS recovery = a + b * sink on the non-gated models (model means
over SAE seeds), 95% t-interval for b (n-2 dof), s_tot = residual SD, band_g = max(0.03, 2 s_tot, h_g),
swing = |b| R with upper end max(|b_lo|, |b_hi|) R, REJECTED checked first.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).parent
FIT_ARMS = ("softmax", "softmax1", "sinklogit", "kbias")
GATED = "gated"


def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    xb, yb = x.mean(), y.mean()
    sxx = ((x - xb) ** 2).sum()
    b = ((x - xb) * (y - yb)).sum() / sxx
    a = yb - b * xb
    res = y - (a + b * x)
    s = np.sqrt((res ** 2).sum() / (n - 2))
    t = stats.t.ppf(0.975, n - 2)
    se_b = s / np.sqrt(sxx)
    return {"a": a, "b": b, "b_ci": [b - t * se_b, b + t * se_b], "s_tot": s, "t": t,
            "sxx": sxx, "xbar": xb, "n": n}


def h_mean(fit, x0):
    return fit["t"] * fit["s_tot"] * np.sqrt(1 / fit["n"] + (x0 - fit["xbar"]) ** 2 / fit["sxx"])


def sink_levels(xs, gap=0.1):
    xs = sorted(xs)
    return 1 + sum(1 for u, v in zip(xs, xs[1:]) if v - u > gap)


def status_rule(fit, R, gated_rows):
    b_lo, b_hi = fit["b_ci"]
    swing_hi = max(abs(b_lo), abs(b_hi)) * R
    out_band = sum(1 for g in gated_rows if not g["in_band"])
    in_band = len(gated_rows) - out_band
    rej = []
    if swing_hi < 0.05:
        rej.append("i")
    if out_band >= 2:
        rej.append("ii")
    if b_lo > 0:
        rej.append("iii")
    if R < 0.15 and "i" in rej:          # D4 range guard (Kasper #17233, Tove #17235)
        rej.remove("i")
    if rej:
        return "REJECTED", rej
    if b_hi < 0 and in_band >= 2 and R >= 0.15:
        return "CONFIRMED", []
    return "UNCONFIRMED", []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="L2_m256_k8")
    ap.add_argument("--protocol", default="d8", choices=["d8", "matched"],
                    help="d8 = every model at step 12000 (primary, Felix #17294); matched = the "
                         "registered first-eval-<=-L* checkpoint (fine-eval rerun if one exists)")
    a = ap.parse_args()
    L_star = float((HERE / "results" / "Lstar.txt").read_text())
    lvb = json.loads((HERE / "results" / "loss_vs_bayes.json").read_text()) \
        if (HERE / "results" / "loss_vs_bayes.json").exists() else {}
    models, floors = [], {"untrained": {}, "embedding": None}
    for arm in FIT_ARMS + (GATED,):
        for rd in sorted((HERE / "runs").glob(f"{arm}_s[0-9]")):
            if a.protocol == "d8":
                fs = [rd / f"measure_{a.tag}_c12000.json"]
            else:
                fine = rd.parent / f"{rd.name}_fine"
                src = fine if fine.exists() else rd
                fs = [f for f in sorted(src.glob(f"measure_{a.tag}_c[0-9]*.json"))
                      if not f.name.endswith("_c12000.json")]
            if not fs or not fs[0].exists():
                continue
            assert len(fs) == 1, fs
            m = json.loads(fs[0].read_text())
            gap = (L_star - m["eval_loss"]) / L_star
            if a.protocol == "matched":
                assert -1e-9 <= gap <= 0.01, (rd.name, m["eval_loss"], L_star)
            lb = lvb.get(f"{rd.name}:{m['ckpt']}", {})
            hist = {h["step"]: h["eval"] for h in
                    json.loads((rd / "log.json").read_text())["history"]}
            plateau_rel = ((hist[8000] - hist[12000]) / hist[12000]
                           if 8000 in hist and 12000 in hist else None)
            models.append({
                "run": rd.name, "arm": m["attn"], "seed": m["seed"], "d": m["d"],
                "gated": m["attn"] == GATED, "in_fit": m["attn"] in FIT_ARMS,
                "sink_mass": m["sink_mass"], "sink_mass_all": m["sink_mass_all"],
                "sink_blk": m["sink_blk"], "dump_mass": m["dump_mass"],
                "trig_mass": m.get("trig_mass"), "noop_mass": m.get("noop_mass"),
                "recovery": m["recovery"], "recovery_sae": m["recovery_sae"],
                "recovery_sd": m["recovery_sd"], "recovery_hungarian": m["recovery_hungarian"],
                "recovery_all_pos": m["recovery_all_pos"], "random_floor": m["random_floor"],
                "cosine_recovery": m["cosine_recovery"], "dead_share": m["dead_share"],
                "eval_loss": m["eval_loss"], "loss_gap_to_L_star": gap,
                "evidence_loss": lb.get("model"), "evidence_bayes": lb.get("bayes"),
                "evidence_share_of_gain": lb.get("share_of_gain"),
                "step": m["ckpt"], "n_params": m["n_params"], "readout_r2": m["readout_r2"],
                "readout_corr": m["readout_corr"],
                "readout_minus_sae": float(np.mean([q["readout_minus_sae"] for q in m["saes"]])),
                "norm_r2": m["norm_r2"], "norm_r2_all_layers": m["norm_r2_all_layers"],
                "gate_r2": m["gate_r2"], "logit_lens_kl": m["logit_lens_kl"],
                "massive_ratio": m["massive"]["max_abs"] / m["massive"]["median_abs"],
                "massive_token_type": m["massive"].get("max_token_type"),
                "recovery_shuffled": float(np.mean([s["mcc_shuffled"] for s in m["saes"]])),
                "measure_file": str(fs[0].relative_to(HERE)),
                "loss_8k": hist.get(8000), "loss_12k": hist.get(12000), "plateau_rel": plateau_rel,
            })
        u = HERE / "runs" / f"{arm}_s0" / f"measure_untrained_{a.tag}.json"
        if u.exists():
            floors["untrained"][arm] = json.loads(u.read_text())["recovery"]
    e = HERE / "runs" / "softmax_s0" / "measure_embed_L0.json"
    if e.exists():
        floors["embedding"] = json.loads(e.read_text())["recovery"]
    man = {"floors": floors}
    fitm = [m for m in models if m["in_fit"]]
    gm = [m for m in models if m["gated"]]
    x = [m["sink_mass"] for m in fitm]
    y = [m["recovery"] for m in fitm]
    fit = ols(x, y)
    R = max(x) - min(x)
    s_sae = float(np.sqrt(np.mean([m["recovery_sd"] ** 2 for m in models])))
    grows = []
    for g in gm:
        pred = fit["a"] + fit["b"] * g["sink_mass"]
        band = max(0.03, 2 * fit["s_tot"], h_mean(fit, g["sink_mass"]))
        grows.append({"run": g["run"], "sink_mass": g["sink_mass"], "recovery": g["recovery"],
                      "pred": pred, "band": band, "residual": g["recovery"] - pred,
                      "in_band": abs(g["recovery"] - pred) <= band})
    status, rej = status_rule(fit, R, grows)
    xs = np.linspace(0, max(x + [g["sink_mass"] for g in gm]), 41)
    line = [{"sink": float(v), "pred": float(fit["a"] + fit["b"] * v),
             "lo": float(fit["a"] + fit["b"] * v - h_mean(fit, v)),
             "hi": float(fit["a"] + fit["b"] * v + h_mean(fit, v))} for v in xs]
    steps = [m["step"] for m in fitm]
    soft = {m["seed"]: m for m in models if m["arm"] == "softmax" and m["seed"] < 3}
    gat = {m["seed"]: m for m in gm}
    pairs = sorted(set(soft) & set(gat))
    outcome = "held" if (len(pairs) == 3 and all(
        gat[s]["recovery"] - soft[s]["recovery"] >= 0.10 and
        gat[s]["sink_mass"] < 0.2 * soft[s]["sink_mass"] for s in pairs) and all(
        m["norm_r2"] < 0.2 for m in models if m["arm"] in ("softmax", GATED))) else "failed"
    fit_loss = None
    if len(fitm) >= 4:
        X = np.column_stack([np.ones(len(fitm)), x, [m["eval_loss"] for m in fitm]])
        yv = np.asarray(y, float)
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        res_ = yv - X @ beta
        dof = len(fitm) - 3
        s2 = (res_ ** 2).sum() / dof
        cov = s2 * np.linalg.inv(X.T @ X)
        tq = stats.t.ppf(0.975, dof)
        se = np.sqrt(np.diag(cov))
        fit_loss = {"a": beta[0], "b": beta[1], "c": beta[2], "dof": dof,
                    "b_ci": [beta[1] - tq * se[1], beta[1] + tq * se[1]],
                    "c_ci": [beta[2] - tq * se[2], beta[2] + tq * se[2]],
                    "b_keeps_sign": bool(np.sign(beta[1]) == np.sign(fit["b"])),
                    "b_ci_excludes_0": bool((beta[1] - tq * se[1]) * (beta[1] + tq * se[1]) > 0)}
    sm = [m for m in models if m["arm"] == "softmax"]
    within = None
    if len(sm) >= 3:
        fw = ols([m["sink_mass"] for m in sm], [m["recovery"] for m in sm])
        rho = stats.spearmanr([m["sink_mass"] for m in sm], [m["recovery"] for m in sm])
        within = {"n": len(sm), "b": fw["b"], "b_ci": fw["b_ci"], "a": fw["a"],
                  "spearman": float(rho.statistic), "spearman_p": float(rho.pvalue),
                  "runs": [m["run"] for m in sm]}
    res = {
        "models": models, "L_star": L_star,
        "lstar_record": json.loads((HERE / "results" / "lstar.json").read_text())
        if (HERE / "results" / "lstar.json").exists() else None,
        "sink_levels_note": "clusters of non-gated sink masses with gaps > 0.1 (PREREG s3)",
        "n_params": {m["arm"]: m["n_params"] for m in models},
        "fit": {"a": fit["a"], "b": fit["b"], "b_ci": fit["b_ci"], "s_tot": fit["s_tot"],
                "s_sae": s_sae, "n": fit["n"], "R": R, "swing": abs(fit["b"]) * R,
                "swing_hi": max(abs(fit["b_ci"][0]), abs(fit["b_ci"][1])) * R,
                "sink_levels": sink_levels(x), "line": line,
                "pearson_steps_sink": float(np.corrcoef(steps, x)[0, 1])},
        "gated": grows, "status": status, "rejected_clauses": rej, "outcome": outcome,
        "pairs": [{"seed": s, "gain": gat[s]["recovery"] - soft[s]["recovery"],
                   "sink_ratio": gat[s]["sink_mass"] / soft[s]["sink_mass"]} for s in pairs],
        "floors": man.get("floors", {}),
        "within_softmax": within,
        "fit_loss": fit_loss,
    }
    res["protocol"] = a.protocol
    out = HERE / "results" / ("results.json" if a.protocol == "d8" else "results_matched.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("status", "rejected_clauses", "outcome")}),
          json.dumps({k: v for k, v in res["fit"].items() if k != "line"}, default=float))


if __name__ == "__main__":
    main()
