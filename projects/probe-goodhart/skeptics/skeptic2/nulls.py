"""skeptic2 null / robustness tests on the probe-goodhart raw data (CPU only).

1. Second-judge DiD: rerun the headline with judge2 (Sonnet) labels instead of judge 1.
2. Undetected-lie rate (judged lie AND probe not flagged), both judges, r0 vs final.
3. Length null: are the lam=1 probe drops explained by answer length? Recall on judged
   lies within word-count tertiles (bins fit on pooled final-round answers of both runs),
   plus a logistic model flag ~ lam + judged_lie + words.
4. Judge agreement (Cohen kappa) per run x round.

Usage: uv run python nulls.py <project_data_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).parent))
import rebuild  # noqa: E402  (own helpers: load_rows, drops, boot_ci, signflip_p)

data = Path(sys.argv[1])
df = rebuild.load_rows(data / "raw")
j2 = pd.read_json(data / "judge2.jsonl", lines=True)
key = ["run", "round", "prompt_id", "sample"]
df = df.merge(j2[key + ["judge2"]], on=key, how="left", validate="one_to_one")
df["judge2_deceptive"] = df["judge2"].map({"DECEPTIVE": 1.0, "HONEST": 0.0})  # OTHER -> NaN
runs = sorted(df.run.unique())
last = int(df["round"].max())
print(f"rows {len(df)}, runs {runs}, judge2 missing {df.judge2.isna().sum()}")

print("\n== 1. headline DiD with judge 2 (lam1 - lam0, r0 -> final, prompt-paired) ==")
for m in ("judge_deceptive", "judge2_deceptive"):
    dt, dc = rebuild.drops(df, "lam1_s0", m), rebuild.drops(df, "lam0_s0", m)
    c = dt.index.intersection(dc.index)
    d = (dt[c] - dc[c]).to_numpy(float)
    lo, hi = rebuild.boot_ci(d)
    print(f"{m:18s} lam1 {dt[c].mean():+.3f}  lam0 {dc[c].mean():+.3f}  DiD {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}] p={rebuild.signflip_p(d):.4f} n={len(c)}")

print("\n== 2. undetected-lie rate = P(judged lie AND not flagged) over all answers ==")
rows = []
for run in runs:
    for rnd in (0, last):
        g = df[(df.run == run) & (df["round"] == rnd)]
        for jl in ("judge_deceptive", "judge2_deceptive"):
            gg = g.dropna(subset=[jl])
            rec = {"run": run, "round": rnd, "judge": jl.split("_")[0], "n": len(gg), "lie_rate": gg[jl].mean()}
            for f in ("flagA", "flagB", "flagA_base"):
                rec[f"undetected_{f}"] = ((gg[jl] == 1) & (gg[f] == 0)).mean()
                rec[f"recall_{f}"] = gg.loc[gg[jl] == 1, f].mean()
            rows.append(rec)
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n== 3. length null at final round ==")
g = df[df["round"] == last].copy()
g["tert"] = pd.qcut(g["resp_words"], 3, labels=["short", "mid", "long"])
lies = g[g.judge_deceptive == 1]
t = lies.groupby(["tert", "run"], observed=True).agg(n=("flagA", "size"), recallA=("flagA", "mean"), recallB=("flagB", "mean")).unstack("run")
print(t.round(3).to_string())
print("word-count tertile edges:", np.round(pd.qcut(g["resp_words"], 3, retbins=True)[1], 1))
X = pd.DataFrame({"lam1": (g.lam > 0).astype(float), "lie": g.judge_deceptive.fillna(0),
                  "words_z": (g.resp_words - g.resp_words.mean()) / g.resp_words.std()})
for f in ("flagA", "flagB"):
    clf = LogisticRegression(C=1e4, max_iter=5000).fit(X, g[f])
    print(f"logit {f} ~ lam1 + lie + words_z: " + ", ".join(f"{k} {v:+.2f}" for k, v in zip(X.columns, clf.coef_[0])))

print("\n== 4. judge 1 vs judge 2 agreement (HONEST/DECEPTIVE rows only) ==")
for (run, rnd), gg in df.dropna(subset=["judge2_deceptive", "judge_deceptive"]).groupby(["run", "round"]):
    print(f"{run} r{rnd}: n={len(gg)} agree={np.mean(gg.judge_deceptive == gg.judge2_deceptive):.2f} kappa={cohen_kappa_score(gg.judge_deceptive, gg.judge2_deceptive):.2f}")
