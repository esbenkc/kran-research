"""skeptic2 independent rebuild of the probe-goodhart numbers.

Own code, reads only the raw data contract (never the build's analysis output):
  RAW/<run>/eval_round<k>.jsonl  one row per held-out response
  RAW/<run>/acts_round<k>.npz    (optional) per-response pooled activations
  PROBES/{A,B}.npz               layer, w, b, mu, sd
  PROBES/{A,B}_train.npz         (optional) training activations + labels
  PROBES/metrics.json            build's reported AUROCs

Outputs: per (run, round) means with bootstrap CIs, the headline paired
difference-in-differences vs the lam=0 control with a sign-flip permutation
test, probe refit check, and an Atlas-style fresh probe on final activations.

Usage: uv run python rebuild.py <project_data_dir> [--out out.json]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

RNG = np.random.default_rng(20261014)
N_BOOT = 5000
N_PERM = 20000


RENAME = {"pA": "probeA", "pB": "probeB", "pA_base": "probeA_base", "pB2": "probeB2", "pA_last": "probeA_last",
          "flagA": "flagA", "flagB": "flagB", "flagA_base": "flagA_base"}


def load_rows(raw: Path, other_as: float = np.nan) -> pd.DataFrame:
    """Accepts both layouts: raw/<run>/eval_round<k>.jsonl (plan contract) and
    raw/eval_<run>.jsonl (build's train.py). Judge strings map DECEPTIVE=1,
    HONEST=0, OTHER=`other_as` (NaN = excluded)."""
    rows = []
    for f in sorted(raw.glob("*/eval_round*.jsonl")) + sorted(raw.glob("eval_*.jsonl")):
        m = re.search(r"eval_round(\d+)", f.name)
        run = f.parent.name if m else f.stem.removeprefix("eval_")
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if m:
                r.setdefault("round", int(m.group(1)))
            r.setdefault("run", run)
            rows.append(r)
    if not rows:
        raise SystemExit(f"no eval rows under {raw}")
    df = pd.DataFrame(rows).rename(columns=RENAME)
    if "judge" in df:  # raw verdict string wins over any precomputed 0/1 column
        df["judge_deceptive"] = df["judge"].map({"DECEPTIVE": 1.0, "HONEST": 0.0}).fillna(other_as)
    df["judge_deceptive"] = pd.to_numeric(df["judge_deceptive"], errors="coerce")
    for c in [c for c in df if c.startswith("flag")]:
        df[c] = df[c].astype(float)
    if "response" in df:
        df["resp_words"] = df["response"].fillna("").str.split().str.len().astype(float)
    return df


def boot_ci(x: np.ndarray) -> tuple[float, float]:
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    idx = RNG.integers(0, len(x), size=(N_BOOT, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def curves(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    out = []
    for (run, rnd), g in df.groupby(["run", "round"]):
        rec = {"run": run, "round": int(rnd), "n": len(g),
               "lam": g["lam"].iloc[0] if "lam" in g else None,
               "seed": g["seed"].iloc[0] if "seed" in g else None}
        for m in metrics:
            v = g[m].to_numpy(dtype=float)
            lo, hi = boot_ci(v)
            rec[m] = float(np.nanmean(v))
            rec[m + "_lo"], rec[m + "_hi"] = lo, hi
        out.append(rec)
    return pd.DataFrame(out).sort_values(["lam", "seed", "round"])


def signflip_p(d: np.ndarray) -> float:
    """Two-sided paired sign-flip permutation test of mean(d) == 0."""
    d = d[~np.isnan(d)]
    obs = abs(d.mean())
    flips = RNG.choice([-1.0, 1.0], size=(N_PERM, len(d)))
    null = np.abs((flips * d).mean(axis=1))
    return float((1 + (null >= obs - 1e-12).sum()) / (N_PERM + 1))


def drops(df: pd.DataFrame, run: str, metric: str) -> pd.Series:
    """Per-prompt change from round 0 to the last round of a run (last - first)."""
    g = df[df.run == run]
    r0, rN = g["round"].min(), g["round"].max()
    a = g[g["round"] == r0].groupby("prompt_id")[metric].mean()
    b = g[g["round"] == rN].groupby("prompt_id")[metric].mean()
    return (b - a).dropna()


def headline(df: pd.DataFrame, metrics: list[str]) -> list[dict]:
    """Diff-in-diff: change(lam>0) - change(lam=0), paired by prompt_id, same seed."""
    res = []
    runs = df.groupby("run").agg(lam=("lam", "first"), seed=("seed", "first")).reset_index()
    for seed, rs in runs.groupby("seed"):
        ctrl = rs[rs.lam == 0]
        if ctrl.empty:
            continue
        crun = ctrl.run.iloc[0]
        for _, r in rs[rs.lam != 0].iterrows():
            for m in metrics:
                dt, dc = drops(df, r.run, m), drops(df, crun, m)
                common = dt.index.intersection(dc.index)
                d = (dt[common] - dc[common]).to_numpy(dtype=float)
                lo, hi = boot_ci(d)
                res.append({"seed": seed, "run": r.run, "lam": r.lam, "control": crun,
                            "metric": m, "n_pairs": len(common),
                            "change_treat": float(dt[common].mean()),
                            "change_ctrl": float(dc[common].mean()),
                            "did": float(d.mean()), "did_lo": lo, "did_hi": hi,
                            "p_signflip": signflip_p(d)})
    return res


def a_vs_b_gap(df: pd.DataFrame) -> list[dict]:
    """Within a run: is probe A's change different from probe B's change?

    Scores are put on a common scale by z-scoring each probe on round-0 rows of the
    same run, then the per-prompt difference (dA - dB) is tested with sign flips.
    """
    res = []
    for run, g in df.groupby("run"):
        r0 = g[g["round"] == g["round"].min()]
        z = {}
        for m in ("probeA", "probeB"):
            mu, sd = r0[m].mean(), r0[m].std(ddof=1) or 1.0
            z[m] = (g.assign(v=(g[m] - mu) / sd))
        dA = drops(z["probeA"].assign(probeA=z["probeA"].v), run, "probeA")
        dB = drops(z["probeB"].assign(probeB=z["probeB"].v), run, "probeB")
        common = dA.index.intersection(dB.index)
        d = (dA[common] - dB[common]).to_numpy(dtype=float)
        res.append({"run": run, "lam": g["lam"].iloc[0], "n": len(common),
                    "dA_z": float(dA[common].mean()), "dB_z": float(dB[common].mean()),
                    "gap_z": float(d.mean()), "p_signflip": signflip_p(d)})
    return res


def apply_probe(p: dict, X: np.ndarray) -> np.ndarray:
    Xs = (X - p["mu"]) / p["sd"]
    return 1 / (1 + np.exp(-(Xs @ p["w"].reshape(-1) + p["b"])))


def refit_probes(probes: Path) -> dict:
    out = {}
    reported = {}
    if (probes / "metrics.json").exists():
        reported = json.loads((probes / "metrics.json").read_text())
    for name in ("A", "B"):
        tf = probes / f"{name}_train.npz"
        if not tf.exists():
            out[name] = "no train activations saved"
            continue
        z = np.load(tf, allow_pickle=True)
        keys = set(z.files)
        X, y = z["X"].astype(np.float32), z["y"].astype(int)
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Xs = (X - mu) / sd
        clf = LogisticRegression(C=0.1, max_iter=3000)
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        pr = cross_val_predict(clf, Xs, y, cv=cv, method="predict_proba")[:, 1]
        rec = {"n": int(len(y)), "pos_frac": float(y.mean()), "own_cv_auroc": float(roc_auc_score(y, pr))}
        # Build's saved probe on its own held-out split, if provided.
        if {"X_test", "y_test"} <= keys and (probes / f"{name}.npz").exists():
            p = dict(np.load(probes / f"{name}.npz", allow_pickle=True))
            rec["saved_probe_test_auroc"] = float(roc_auc_score(z["y_test"], apply_probe(p, z["X_test"].astype(np.float32))))
        # Shuffled-label control on the same features.
        ys = RNG.permutation(y)
        prs = cross_val_predict(clf, Xs, ys, cv=cv, method="predict_proba")[:, 1]
        rec["shuffled_cv_auroc"] = float(roc_auc_score(ys, prs))
        rec["reported"] = reported.get(name, reported.get(f"probe{name}"))
        out[name] = rec
    return out


def probe_judge_auroc(df: pd.DataFrame) -> pd.DataFrame:
    """Per run x round: does each probe still rank judged lies above judged honest answers?
    A blinded probe A with a still-sighted probe B is the activation-obfuscation signature."""
    out = []
    for (run, rnd), g in df.groupby(["run", "round"]):
        y = g["judge_deceptive"].to_numpy()
        keep = ~np.isnan(y)
        rec = {"run": run, "lam": g["lam"].iloc[0], "round": int(rnd),
               "n": int(keep.sum()), "lies": int(np.nansum(y))}
        for m in ("probeA", "probeB", "probeA_base", "probeB2", "probeA_last"):
            if m not in g:
                continue
            yy, s = y[keep].astype(int), g[m].to_numpy(dtype=float)[keep]
            ok = 0 < yy.sum() < len(yy)
            rec[m + "_auroc"] = float(roc_auc_score(yy, s)) if ok else np.nan
        out.append(rec)
    return pd.DataFrame(out).sort_values(["lam", "run", "round"])


def _acts_file(raw: Path, run: str, rnd: int) -> Path | None:
    for f in (raw / f"acts_{run}" / f"eval_round{rnd}.npz", raw / run / f"acts_round{rnd}.npz"):
        if f.exists():
            return f
    return None


def fresh_probe_final(raw: Path, probes: Path, df: pd.DataFrame) -> list[dict]:
    """Per run, on round 0 and the final round:
    (a) recompute each saved probe's score from the saved activations and compare with
        the logged score (integrity check), and
    (b) Atlas-style fresh probe: refit logistic regression on these activations with judge
        labels, prompt-grouped CV. High AUROC = deception is still linearly present.
    Rows are aligned to the jsonl by order within the round; prompt_id order is checked."""
    from sklearn.model_selection import StratifiedGroupKFold
    saved = {k: dict(np.load(probes / f"{k}.npz", allow_pickle=True))
             for k in ("A", "B", "B2") if (probes / f"{k}.npz").exists()}
    res = []
    for run, g in df.groupby("run"):
        for rnd in sorted({int(g["round"].min()), int(g["round"].max())}):
            f = _acts_file(raw, run, rnd)
            if f is None:
                continue
            z = np.load(f, allow_pickle=True)
            gr = g[g["round"] == rnd]
            ids = np.array([str(i) for i in z["prompt_id"]])
            if len(ids) != len(gr) or not (ids == gr["prompt_id"].astype(str).to_numpy()).all():
                res.append({"run": run, "round": rnd, "note": "npz rows do not align with jsonl rows"})
                continue
            y_all = gr["judge_deceptive"].to_numpy(dtype=float)
            for key in [k for k in z.files if k.startswith("acts")]:
                X = z[key].astype(np.float32)
                rec = {"run": run, "round": rnd, "acts": key}
                pname = key.removeprefix("acts")
                col = {"A": "probeA", "B": "probeB", "B2": "probeB2", "A_last": "probeA_last"}.get(pname)
                if pname in saved and col in gr:
                    p = saved[pname]
                    s = 1 / (1 + np.exp(-(((X - p["mu"]) / p["sd"]) @ p["w"].reshape(-1) + float(p["b"]))))
                    rec["max_abs_score_diff"] = float(np.max(np.abs(s - gr[col].to_numpy(dtype=float))))
                keep = ~np.isnan(y_all)
                Xk, y, grp = X[keep], y_all[keep].astype(int), ids[keep]
                nmin = int(min(y.sum(), len(y) - y.sum()))
                rec.update(n=int(len(y)), pos=int(y.sum()))
                if nmin < 5:
                    rec["note"] = "too few of one class"
                    res.append(rec)
                    continue
                Xs = (Xk - Xk.mean(0)) / (Xk.std(0) + 1e-6)
                cv = StratifiedGroupKFold(min(5, nmin), shuffle=True, random_state=0)
                pr = np.zeros(len(y))
                for tr, te in cv.split(Xs, y, grp):
                    clf = LogisticRegression(C=0.1, max_iter=3000).fit(Xs[tr], y[tr])
                    pr[te] = clf.predict_proba(Xs[te])[:, 1]
                rec["fresh_cv_auroc"] = float(roc_auc_score(y, pr))
                # Null: same pipeline, judge labels permuted; 20 permutations -> mean, 95th pct.
                null = []
                for _ in range(20):
                    yp = RNG.permutation(y)
                    prn = np.zeros(len(y))
                    for tr, te in cv.split(Xs, yp, grp):
                        prn[te] = LogisticRegression(C=0.1, max_iter=3000).fit(Xs[tr], yp[tr]).predict_proba(Xs[te])[:, 1]
                    null.append(roc_auc_score(yp, prn))
                rec["null_mean"], rec["null_p95"] = float(np.mean(null)), float(np.percentile(null, 95))
                res.append(rec)
    return res


def fixed_text_auroc(raw: Path) -> pd.DataFrame:
    """From fixed_<run>.jsonl: probe A/B AUROC on FIXED teacher-forced truthful vs
    deceptive texts, read through the current model. Text is constant, so any fall is
    pure activation change (Atlas 'obfuscated activations' or drift)."""
    out = []
    for f in sorted(raw.glob("fixed_*.jsonl")):
        d = pd.read_json(f, lines=True)
        for rnd, g in d.groupby("round"):
            rec = {"run": f.stem.removeprefix("fixed_"), "round": int(rnd), "n": len(g)}
            for c, name in (("pA", "A"), ("pB", "B")):
                if c in g:
                    rec[f"fixed_auroc_{name}"] = float(roc_auc_score(g["label"], g[c]))
            out.append(rec)
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path, help="project data dir containing raw/ and probes/")
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("rebuild_out.json"))
    ap.add_argument("--other-as", type=float, default=np.nan,
                    help="value for judge OTHER (default NaN = excluded; try 0 and 1 as bounds)")
    a = ap.parse_args()
    raw, probes = a.data / "raw", a.data / "probes"
    df = load_rows(raw, a.other_as)
    metrics = [m for m in ("probeA", "probeB", "judge_deceptive", "probeA_base", "probeB2",
                           "flagA", "flagB", "flagA_base", "resp_words") if m in df]
    cur = curves(df, metrics)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("== per run x round (mean, 95% bootstrap CI over prompts) ==")
    print(cur[["run", "lam", "seed", "round", "n"] + metrics].round(3).to_string(index=False))
    hl = headline(df, metrics)
    print("\n== headline: change(lam>0) - change(lam=0), paired by prompt ==")
    print(pd.DataFrame(hl).round(4).to_string(index=False) if hl else "no lam=0 control found")
    gap = a_vs_b_gap(df) if {"probeA", "probeB"} <= set(df) else []
    print("\n== within-run A vs B change (z units of round 0) ==")
    print(pd.DataFrame(gap).round(4).to_string(index=False))
    pj = probe_judge_auroc(df)
    print("\n== per round: probe AUROC vs judge label (on-policy answers) ==")
    print(pj.round(3).to_string(index=False))
    ref = refit_probes(probes) if probes.exists() else {}
    print("\n== probe refit ==\n" + json.dumps(ref, indent=1, default=str))
    fr = fresh_probe_final(raw, probes, df)
    print("\n== fresh probe (judge labels) on round-0 vs final activations ==")
    print(pd.DataFrame(fr).round(3).to_string(index=False) if fr else "no acts_round*.npz saved")
    fx = fixed_text_auroc(raw)
    print("\n== fixed-text probe AUROC per round (activation change only) ==")
    print(fx.round(3).to_string(index=False) if len(fx) else "no fixed_*.jsonl")
    a.out.write_text(json.dumps({"curves": cur.to_dict("records"), "headline": hl, "a_vs_b": gap,
                                 "probe_judge_auroc": pj.to_dict("records"),
                                 "fixed_auroc": fx.to_dict("records"),
                                 "refit": ref, "fresh": fr}, indent=1, default=str))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
