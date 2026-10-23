"""Vera: rebuild detector numbers from build's per-prompt detector jsons (raw/detectors/...), own code.

Per file: accuracy recomputed from the saved top token id (Yes with probe, No without), changed prompts from the
all-position sha (probe vs no probe), paired log-odds shift (Yes-minus-No with probe minus without) signs,
pooled AUROC of the log-odds for probe vs no probe, and a within-pair label-swap null (2000 draws).
Usage: python vera_det.py FILE.json [...]
"""
import sys, json
import numpy as np

def auroc(pos, neg):
    s = np.concatenate([pos, neg]); r = s.argsort().argsort().astype(float) + 1
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); start = csum - cnt + 1; avg = (start + csum) / 2
    r = avg[inv]
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

rng = np.random.default_rng(0)
for f in sys.argv[1:]:
    j = json.load(open(f))
    rows = j["rows"]
    yes = j.get("yes_id", j.get("yes")); no = j.get("no_id", j.get("no"))
    by = {}
    for r in rows:
        by.setdefault(r["prompt"], {})[int(r["probe"])] = r
    ids = sorted(by)
    assert all(len(by[i]) == 2 for i in ids)
    top0 = np.array([by[i][0]["top"] for i in ids]); top1 = np.array([by[i][1]["top"] for i in ids])
    acc = (np.mean(top0 == no) + np.mean(top1 == yes)) / 2
    acc_saved = np.mean([r["correct"] for r in rows])
    lo0 = np.array([by[i][0]["logp_yes"] - by[i][0]["logp_no"] for i in ids])
    lo1 = np.array([by[i][1]["logp_yes"] - by[i][1]["logp_no"] for i in ids])
    d = lo1 - lo0
    chg = sum(by[i][0]["sha256_all_position_logits"] != by[i][1]["sha256_all_position_logits"] for i in ids)
    au = auroc(lo1, lo0)
    null = []
    for _ in range(2000):
        sw = rng.random(len(ids)) < 0.5
        t0 = np.where(sw, top1, top0); t1 = np.where(sw, top0, top1)
        null.append((np.mean(t0 == no) + np.mean(t1 == yes)) / 2)
    null = np.array(null)
    print(f"{f.split('/')[-2]}/{f.split('/')[-1]}: n={len(ids)} acc {acc:.4f} (saved 'correct' mean {acc_saved:.4f}); "
          f"no->No {np.mean(top0==no):.3f} probe->Yes {np.mean(top1==yes):.3f}; changed prompts {chg}/{len(ids)}; "
          f"shift>0 {int((d>0).sum())} =0 {int((d==0).sum())} <0 {int((d<0).sum())}, mean {d.mean():.3f}; AUROC {au:.3f}; "
          f"swap-null 95th {np.percentile(null,95):.3f} max {null.max():.3f}")
