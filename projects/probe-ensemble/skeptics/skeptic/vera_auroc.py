"""Vera: held-out probe AUROC per run/round from per-answer scores in saved eval rows.

AUROC of score vs judge_deceptive (1 = DECEPTIVE, 0 = HONEST; rows with label None dropped),
prompt-cluster bootstrap CI, shuffled-label null. Also the ensemble/probe-A margin (mean score on
deceptive minus honest answers).
usage: python vera_auroc.py --score pattn [--score2 pens pA] eval_*.jsonl
"""
import argparse, json, collections
import numpy as np
from vera_stats import auroc  # my own implementation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--score", default="pattn")
    ap.add_argument("--margin", nargs="*", default=[])
    ap.add_argument("--label", default="judge_deceptive")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    out = {}
    for f in a.files:
        rows = [json.loads(l) for l in open(f) if l.strip()]
        by = collections.defaultdict(list)
        for r in rows:
            if r.get(a.label) is None or r.get(a.score) is None or r.get("judge") == "OTHER":
                continue
            by[int(r["round"])].append(r)
        run = rows[0]["run"]
        for rd in sorted(by):
            R = by[rd]
            y = np.array([int(r[a.label]) for r in R]); s = np.array([float(r[a.score]) for r in R])
            if y.min() == y.max():
                print(run, rd, "one class only"); continue
            au = auroc(s, y)
            pids = np.array([r["prompt_id"] for r in R]); up = np.unique(pids)
            idx = {p: np.where(pids == p)[0] for p in up}
            bs = []
            for _ in range(1000):
                pick = np.concatenate([idx[p] for p in rng.choice(up, len(up))])
                if y[pick].min() != y[pick].max():
                    bs.append(auroc(s[pick], y[pick]))
            null = [auroc(s, rng.permutation(y)) for _ in range(200)]
            line = (f"{run:<12} r{rd}  n={len(y)} dec={y.mean():.3f}  {a.score} AUROC {au:.3f} "
                    f"CI [{np.percentile(bs,2.5):.3f},{np.percentile(bs,97.5):.3f}]  null99 {np.percentile(null,99):.3f}")
            for m in a.margin:
                v = np.array([float(r[m]) for r in R])
                line += f"  margin[{m}] {v[y==1].mean()-v[y==0].mean():+.3f} (hon {v[y==0].mean():.3f})"
            print(line)
            out[f"{run}_r{rd}"] = au
    json.dump(out, open(f"vera_auroc_{a.score}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
