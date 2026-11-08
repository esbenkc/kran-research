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
    ap.add_argument("--manifest", default=str(HERE / "results" / "manifest.json"))
    a = ap.parse_args()
    man = json.loads(Path(a.manifest).read_text())
    L_star = man["L_star"]
    models = []
    for row in man["models"]:
        m = json.loads((HERE / "runs" / row["run"] / f"measure_{row['tag'] or a.tag}.json").read_text())
        gap = (L_star - m["eval_loss"]) / L_star
        assert -1e-9 <= gap <= 0.01, (row["run"], m["eval_loss"], L_star)
        models.append({
            "run": row["run"], "arm": m["attn"], "seed": m["seed"], "d": m["d"],
            "gated": m["attn"] == GATED, "in_fit": m["attn"] in FIT_ARMS,
            "sink_mass": m["sink_mass"], "sink_mass_all": m["sink_mass_all"],
            "dump_mass": m["dump_mass"], "recovery": m["recovery"],
            "recovery_sae": m["recovery_sae"], "recovery_sd": m["recovery_sd"],
            "cosine_recovery": m["cosine_recovery"], "eval_loss": m["eval_loss"],
            "loss_gap_to_L_star": gap, "excess_bayes": m["eval_loss"] - man["bayes_loss"],
            "step": m["ckpt"], "readout_r2": m["readout_r2"], "norm_r2": m["norm_r2"],
            "gate_r2": m["gate_r2"], "logit_lens_kl": m["logit_lens_kl"],
            "massive_ratio": m["massive"]["max_abs"] / m["massive"]["median_abs"],
            "recovery_shuffled": float(np.mean([s["mcc_shuffled"] for s in m["saes"]])),
        })
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
    soft = {m["seed"]: m for m in models if m["arm"] == "softmax"}
    gat = {m["seed"]: m for m in gm}
    pairs = sorted(set(soft) & set(gat))
    outcome = "held" if (len(pairs) == 3 and all(
        gat[s]["recovery"] - soft[s]["recovery"] >= 0.10 and
        gat[s]["sink_mass"] < 0.2 * soft[s]["sink_mass"] for s in pairs) and all(
        m["norm_r2"] < 0.2 for m in models if m["arm"] in ("softmax", GATED))) else "failed"
    res = {
        "models": models, "L_star": L_star, "bayes_loss": man["bayes_loss"],
        "fit": {"a": fit["a"], "b": fit["b"], "b_ci": fit["b_ci"], "s_tot": fit["s_tot"],
                "s_sae": s_sae, "n": fit["n"], "R": R, "swing": abs(fit["b"]) * R,
                "swing_hi": max(abs(fit["b_ci"][0]), abs(fit["b_ci"][1])) * R,
                "sink_levels": sink_levels(x), "line": line,
                "pearson_steps_sink": float(np.corrcoef(steps, x)[0, 1])},
        "gated": grows, "status": status, "rejected_clauses": rej, "outcome": outcome,
        "pairs": [{"seed": s, "gain": gat[s]["recovery"] - soft[s]["recovery"],
                   "sink_ratio": gat[s]["sink_mass"] / soft[s]["sink_mass"]} for s in pairs],
        "floors": man.get("floors", {}),
    }
    out = HERE / "results" / "results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("status", "rejected_clauses", "outcome")}),
          json.dumps({k: v for k, v in res["fit"].items() if k != "line"}, default=float))


if __name__ == "__main__":
    main()
