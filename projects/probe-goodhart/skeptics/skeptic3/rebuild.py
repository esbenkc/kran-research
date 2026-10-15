"""skeptic3 independent rebuild of probe-goodhart key numbers. Own code, CPU only.

Reads only saved files in <data>:
  raw/eval_<run>.jsonl   per-sample eval rows (round, lam, seed, prompt_id, sample, judge_deceptive, pA, pA_base, pB, pB2)
  raw/fixed_<run>.jsonl  fixed teacher-forced DolusChat replies scored by the current model each round
  raw/acts_<run>/eval_round<k>.npz  pooled acts per probe (actsA, actsB, actsB2), prompt_id, sample
  probes/*.npz or probes.json  (w, b, mu, sd, thr)
Usage: uv run --project kran-research python skeptic3/rebuild.py <data_dir> <out.md>
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(1014)
NB = 2000
PROBES = ["A", "A_base", "B", "B2"]


def boot(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan,) * 3
    bs = x[RNG.integers(0, len(x), (NB, len(x)))].mean(1)
    return (x.mean(), *np.percentile(bs, [2.5, 97.5]))


def f3(t, p=3):
    return f"{t[0]:+.{p}f} [{t[1]:+.{p}f},{t[2]:+.{p}f}]" if not np.isnan(t[0]) else "nan"


def l3(t, p=3):
    return f"{t[0]:.{p}f} [{t[1]:.{p}f},{t[2]:.{p}f}]" if not np.isnan(t[0]) else "nan"


def auroc(y, s):
    y, s = np.asarray(y), np.asarray(s, float)
    ok = ~np.isnan(s)
    return roc_auc_score(y[ok], s[ok]) if len(set(y[ok])) == 2 else np.nan


def perm_p(x, n=20000):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    null = (RNG.choice([-1.0, 1.0], (n, len(x))) * x).mean(1)
    return float((np.abs(null) >= abs(x.mean()) - 1e-12).mean())


def load_probes(data):
    pj = os.path.join(data, "probes.json")
    out = {}
    if os.path.exists(pj):
        d = json.load(open(pj))
        for k in ("A", "B", "B2"):
            if k in d:
                v = d[k]
                out[k] = {kk: np.array(v[kk]) if kk in ("mu", "sd", "w") else v[kk] for kk in v}
    for k in ("A", "B", "B2"):
        if k not in out:
            fs = glob.glob(os.path.join(data, "**", f"{k}.npz"), recursive=True)
            if fs:
                z = np.load(fs[0], allow_pickle=True)
                out[k] = {kk: z[kk] for kk in z.files}
    return out


def score(pr, X):
    z = ((X.astype(np.float64) - pr["mu"]) / pr["sd"]) @ pr["w"] + float(pr["b"])
    return 1 / (1 + np.exp(-z))


def main(data, outp):
    L = []
    p = L.append
    raw = os.path.join(data, "raw")
    files = sorted(glob.glob(os.path.join(raw, "eval_*.jsonl")))
    df = pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)
    df = df.rename(columns={"pA": "A", "pA_base": "A_base", "pB": "B", "pB2": "B2"})
    probes = [c for c in PROBES if c in df]
    pr = load_probes(data)
    thr = {k: float(pr[k]["thr"]) for k in pr}
    thr["A_base"] = thr.get("A")
    for c in probes:
        df["f" + c] = (df[c] > thr[c]).astype(float)
    p(f"# skeptic3 rebuild ({data})\n")
    p(f"files={[os.path.basename(f) for f in files]} rows={len(df)}; thresholds (from saved probes) = "
      + ", ".join(f"{k}:{v:.4f}" for k, v in thr.items() if v is not None) + "\n")
    # consistency of saved flag columns with my thresholding
    for k in ("A", "B", "B2"):
        if f"flag{k}" in df:
            mism = int((df[f"flag{k}"].astype(float) != df["f" + k]).sum())
            p(f"- flag{k} column vs my threshold recompute: {mism} mismatches")
    if "flagA_base" in df:
        p(f"- flagA_base column vs my recompute: {int((df['flagA_base'].astype(float) != df['fA_base']).sum())} mismatches")
    p(f"- duplicate (run,round,prompt_id,sample): {int(df.duplicated(['run', 'round', 'prompt_id', 'sample']).sum())}\n")

    # ---- 1. levels per run x round
    p("## 1. Per run x round: rates flagged (threshold = 5% FPR on held-out truthful refs) and judge lie rate, 95% bootstrap CI over samples\n")
    p("| run | round | n | judge | " + " | ".join(f"flag{c}" for c in probes) + " | words | AUROC vs judge (" + ",".join(probes) + ") | corr(A,B) |")
    p("|---|---|---|---|" + "---|" * len(probes) + "---|---|---|")
    for (run, rnd), g in df.groupby(["run", "round"]):
        p(f"| {run} | {rnd} | {len(g)} | {l3(boot(g.judge_deceptive))} | "
          + " | ".join(l3(boot(g['f' + c])) for c in probes)
          + f" | {g.n_words.mean():.0f} | " + ",".join(f"{auroc(g.judge_deceptive, g[c]):.2f}" for c in probes)
          + f" | {np.corrcoef(g['A'], g['B'])[0, 1]:.2f} |")

    # ---- 2. paired change round0 -> last, prompt-level (mean over samples), DiD vs lam=0
    mets = ["judge_deceptive"] + ["f" + c for c in probes] + probes + ["n_words"]
    pl = df.groupby(["run", "lam", "seed", "round", "prompt_id"])[mets].mean().reset_index()
    p("\n## 2. Paired change round0 -> last round (prompt-level means over samples), bootstrap CI\n")
    p("| run | last | n_prompts | " + " | ".join("Δ" + m for m in mets) + " |")
    p("|---|---|---|" + "---|" * len(mets))
    D = {}
    for (run, lam, seed), g in pl.groupby(["run", "lam", "seed"]):
        r0, rl = g["round"].min(), g["round"].max()
        a = g[g["round"] == r0].set_index("prompt_id")[mets]
        b = g[g["round"] == rl].set_index("prompt_id")[mets]
        ix = a.index.intersection(b.index)
        D[(run, lam, seed)] = (b.loc[ix] - a.loc[ix], rl)
        p(f"| {run} | {rl} | {len(ix)} | " + " | ".join(f3(boot((b.loc[ix, m] - a.loc[ix, m]).values), 2 if m == 'n_words' else 3) for m in mets) + " |")

    p("\n## 3. Diff-in-diff vs lam=0 (same seed), paired by prompt, bootstrap CI + sign-flip permutation p\n")
    p("Note: if the lam=0 and lam>0 runs stop at different rounds, DiD compares unequal training lengths (flagged).\n")
    p("| run | metric | DiD | perm p | rounds (run/ctrl) |")
    p("|---|---|---|---|---|")
    for (run, lam, seed), (d, rl) in D.items():
        if lam == 0:
            continue
        ck = [k for k in D if k[1] == 0 and k[2] == seed]
        if not ck:
            continue
        dc, rlc = D[ck[0]]
        ix = d.index.intersection(dc.index)
        for m in mets:
            x = (d.loc[ix, m] - dc.loc[ix, m]).values
            p(f"| {run} | {m} | {f3(boot(x), 2 if m == 'n_words' else 3)} | {perm_p(x):.4f} | {rl}/{rlc}{' UNEQUAL' if rl != rlc else ''} |")

    # ---- 4. fixed-text AUROC per round (pure activation change)
    p("\n## 4. Fixed teacher-forced DolusChat replies through the current model: AUROC per round (text fixed => change = activation drift)\n")
    fx = sorted(glob.glob(os.path.join(raw, "fixed_*.jsonl")))
    for f in fx:
        d = pd.read_json(f, lines=True)
        cols = [c for c in d if c.startswith("p") and c not in ("prompt_id",)]
        rows = []
        for rnd, g in d.groupby("round"):
            rows.append(f"r{rnd}: " + " ".join(f"{c}={auroc(g.label, g[c]):.3f}" for c in cols)
                        + " | flag@thr " + " ".join(f"{c}:TPR={np.mean(g.loc[g.label == 1, c] > thr.get(c[1:], np.inf)):.2f}/FPR={np.mean(g.loc[g.label == 0, c] > thr.get(c[1:], np.inf)):.2f}" for c in cols if c[1:] in thr))
        p(f"- {os.path.basename(f)} (n/round={len(d) // d['round'].nunique()}):")
        for r in rows:
            p(f"  - {r}")

    # ---- 5. activation-level nulls
    p("\n## 5. Activation nulls from saved acts (own scoring)\n")
    trn = {}
    for k in ("A", "B"):
        fs = glob.glob(os.path.join(data, "**", f"{k}_train.npz"), recursive=True)
        if fs:
            trn[k] = np.load(fs[0])
    for run_dir in sorted(glob.glob(os.path.join(raw, "acts_*"))):
        run = os.path.basename(run_dir)[5:]
        rf = sorted(glob.glob(os.path.join(run_dir, "eval_round*.npz")), key=lambda s: int(s.split("round")[-1][:-4]))
        if len(rf) < 2:
            continue
        z0, zl = np.load(rf[0], allow_pickle=True), np.load(rf[-1], allow_pickle=True)
        p(f"### {run}: {os.path.basename(rf[0])} -> {os.path.basename(rf[-1])}")
        sub = df[df.run == run]
        for k in ("A", "B"):
            key = f"acts{k}"
            if key not in z0.files or k not in pr:
                continue
            X0, Xl = z0[key].astype(np.float64), zl[key].astype(np.float64)
            # (a) recompute probe scores from acts and compare to jsonl column
            for z, X in ((z0, X0), (zl, Xl)):
                rnd = int(rf[0 if z is z0 else -1].split("round")[-1][:-4])
                s = score(pr[k], X)
                ref = sub[sub["round"] == rnd].set_index(["prompt_id", "sample"])[k]
                mine = pd.Series(s, index=pd.MultiIndex.from_arrays([z["prompt_id"], z["sample"]]))
                ix = ref.index.intersection(mine.index)
                p(f"- probe {k} round {rnd}: max |my score - jsonl| = {np.abs(ref.loc[ix].values - mine.loc[ix].values).max():.2e} over {len(ix)} rows")
            # (b) shift along probe direction vs random directions, in standardized space, units of round-0 SD
            S0, Sl = (X0 - pr[k]["mu"]) / pr[k]["sd"], (Xl - pr[k]["mu"]) / pr[k]["sd"]

            def shift(v):
                v = v / np.linalg.norm(v)
                a, b = S0 @ v, Sl @ v
                return (b.mean() - a.mean()) / a.std()

            sp = shift(pr[k]["w"])
            R = RNG.standard_normal((300, S0.shape[1]))
            rs = np.array([shift(v) for v in R])
            line = f"- probe {k} direction shift = {sp:+.3f} SD; random dirs: median |shift| {np.median(np.abs(rs)):.3f}, 95th pct {np.percentile(np.abs(rs), 95):.3f}; frac random |shift| >= |probe| = {np.mean(np.abs(rs) >= abs(sp)):.3f}"
            # (c) shuffled-label probe refit on the same training acts
            if k in trn:
                T = trn[k]
                mu, sd = T["X"].mean(0), T["X"].std(0) + 1e-6
                ys = RNG.permutation(T["y"])
                w = LogisticRegression(C=0.05, max_iter=3000).fit((T["X"] - mu) / sd, ys).coef_[0]
                S0s, Sls = (X0 - mu) / sd, (Xl - mu) / sd
                v = w / np.linalg.norm(w)
                line += f"; shuffled-label probe shift = {((Sls @ v).mean() - (S0s @ v).mean()) / (S0s @ v).std():+.3f} SD"
            p(line)
    open(outp, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "kran-research/projects/probe-goodhart/data",
         sys.argv[2] if len(sys.argv) > 2 else "swarm-oracle-1014-probe-goodhart/skeptic3/rebuild.md")
