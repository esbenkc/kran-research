"""Vera: compare my tail rows (out/<name>.json) with build rows (any list of [prompt, probe, top1, logpA, logpB])."""
import json, sys, numpy as np
def cmp(mine_file, build_rows, label):
    m = {(r[0], r[1]): r for r in json.load(open(mine_file))["rows"]}
    g = {(r[0], r[1]): r for r in build_rows}
    k = [x for x in g if x in m]
    top = np.mean([m[x][2] == g[x][2] for x in k])
    lo = np.array([m[x][3] - m[x][4] for x in k]); lg = np.array([g[x][3] - g[x][4] for x in k])
    d = np.abs(lo - lg)
    print(f"{label}: n={len(k)} top1 agree {top:.4f} |dLO| median {np.median(d):.3f} p95 {np.percentile(d,95):.3f} max {d.max():.3f} corr {np.corrcoef(lo,lg)[0,1]:.4f}")
    return dict(top1_agree=float(top), dLO_median=float(np.median(d)), dLO_max=float(d.max()))
if __name__ == "__main__":
    G = json.load(open(sys.argv[2]))["rows"][sys.argv[3]]
    cmp(sys.argv[1], G, sys.argv[3])
