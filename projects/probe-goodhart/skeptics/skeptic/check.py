"""skeptic's independent checker. Reads only raw eval jsonl (+ optional activation npz).

Usage: uv run python check.py <raw_dir> [--probes <probes_dir>]
raw_dir = kran-research/projects/probe-goodhart/data/raw (contains <run>/eval_round<k>.jsonl)
"""
import argparse, glob, json, os, sys
from collections import defaultdict
import numpy as np

rng = np.random.default_rng(12345)


def auroc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann-Whitney with tie handling
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):
        m = allv == v
        ranks[m] = ranks[m].mean()
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def boot_ci(x, n=2000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (float("nan"),) * 2
    b = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def load(raw_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(raw_dir, "*", "eval_round*.jsonl"))):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line); r["_file"] = f; rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir")
    a = ap.parse_args()
    rows = load(a.raw_dir)
    print(f"rows={len(rows)}")
    # integrity
    keys = defaultdict(int)
    for r in rows:
        keys[(r["run"], r["round"], r["prompt_id"])] += 1
    dups = sum(1 for v in keys.values() if v > 1)
    print(f"duplicate (run,round,prompt) keys: {dups}")
    bad = [r for r in rows if r.get("judge_deceptive") not in (0, 1)]
    print(f"rows with judge_deceptive not 0/1: {len(bad)}")

    g = defaultdict(list)
    for r in rows:
        g[(r["lam"], r.get("seed", 0), r["run"], r["round"])].append(r)
    # prompts per round consistent?
    pset = defaultdict(set)
    for (lam, seed, run, rd), rs in g.items():
        pset[run].add(frozenset(x["prompt_id"] for x in rs))
    for run, s in pset.items():
        print(f"run {run}: distinct held-out prompt sets across rounds = {len(s)}")

    print("\nlam seed run round  n  probeA[ci]  probeB[ci]  judge[ci]  AUROC_A  AUROC_B")
    table = []
    for k in sorted(g):
        rs = g[k]
        A = np.array([x["probeA"] for x in rs], float)
        B = np.array([x["probeB"] for x in rs], float)
        J = np.array([x["judge_deceptive"] for x in rs], float)
        ca, cb, cj = boot_ci(A), boot_ci(B), boot_ci(J)
        aA, aB = auroc(A, J), auroc(B, J)
        table.append(dict(lam=k[0], seed=k[1], run=k[2], round=k[3], n=len(rs), A=A.mean(), B=B.mean(), J=J.mean(), aucA=aA, aucB=aB))
        print(f"{k[0]} {k[1]} {k[2]} {k[3]:>2} {len(rs):>3} {A.mean():.3f}[{ca[0]:.3f},{ca[1]:.3f}] {B.mean():.3f}[{cb[0]:.3f},{cb[1]:.3f}] {J.mean():.3f}[{cj[0]:.3f},{cj[1]:.3f}] {aA:.3f} {aB:.3f}")

    # first vs last round deltas per run, with paired permutation (same prompts) where possible
    print("\nfirst->last round deltas (paired by prompt_id), sign-flip permutation p (two-sided)")
    byrun = defaultdict(dict)
    for k, rs in g.items():
        byrun[k[2]][k[3]] = {x["prompt_id"]: x for x in rs}
    for run, rounds in sorted(byrun.items()):
        r0, r1 = min(rounds), max(rounds)
        common = sorted(set(rounds[r0]) & set(rounds[r1]))
        out = [f"run {run} r{r0}->r{r1} n_paired={len(common)}"]
        for m in ("probeA", "probeB", "judge_deceptive"):
            d = np.array([rounds[r1][p][m] - rounds[r0][p][m] for p in common], float)
            if len(d) == 0:
                continue
            obs = d.mean()
            flips = rng.choice([-1, 1], size=(10000, len(d)))
            null = (flips * d).mean(1)
            p = (np.abs(null) >= abs(obs) - 1e-12).mean()
            out.append(f"{m}: d={obs:+.3f} p={p:.4f}")
        print("  ".join(out))
    json.dump(table, open(os.path.join(os.path.dirname(__file__), "skeptic_table.json"), "w"), indent=1, default=float)
    did(rows)


def did(rows, thr=None):
    """Headline: DiD = drop(lam>0) - drop(lam=0), round0 -> final, paired by prompt_id, bootstrap over prompts.
    drop = mean(round0) - mean(final). thr: optional {metric: threshold} to compute flag rates."""
    by = defaultdict(lambda: defaultdict(dict))  # (lam,seed) -> round -> pid -> row
    for r in rows:
        by[(float(r["lam"]), r.get("seed", 0))][r["round"]][r["prompt_id"]] = r
    metrics = [m for m in ("probeA", "probeA_base", "probeB", "probeB2", "judge_deceptive") if m in rows[0]]
    print("\nDiD headline (drop(lam) - drop(lam=0)), bootstrap 95% CI over prompts")
    for seed in sorted({k[1] for k in by}):
        ctrl = by.get((0.0, seed))
        if not ctrl:
            print(f"seed {seed}: no lam=0 control"); continue
        for lam in sorted(l for (l, s) in by if s == seed and l > 0):
            tr = by[(lam, seed)]
            rf_c, rf_t = max(ctrl), max(tr)
            pids = sorted(set(ctrl[0]) & set(ctrl[rf_c]) & set(tr[0]) & set(tr[rf_t]))
            line = [f"seed {seed} lam {lam} (ctrl final r{rf_c}, treat final r{rf_t}, n={len(pids)})"]
            for m in metrics:
                def val(r):
                    v = float(r[m])
                    return float(v >= thr[m]) if thr and m in thr else v
                dc = np.array([val(ctrl[0][p]) - val(ctrl[rf_c][p]) for p in pids])
                dt = np.array([val(tr[0][p]) - val(tr[rf_t][p]) for p in pids])
                d = dt - dc
                idx = rng.integers(0, len(d), size=(5000, len(d)))
                b = d[idx].mean(1)
                line.append(f"{m}: dropT={dt.mean():+.3f} dropC={dc.mean():+.3f} DiD={d.mean():+.3f} [{np.percentile(b,2.5):+.3f},{np.percentile(b,97.5):+.3f}]")
            print("\n  ".join(line))


if __name__ == "__main__":
    main()
