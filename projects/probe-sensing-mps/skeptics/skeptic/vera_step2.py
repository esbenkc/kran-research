"""Rebuild Step-2 detector numbers from build's per-case json (whole-graph or cached runs) + label null.

acc = mean over cases of [top token over full vocab == Yes if probe else No]; both_correct per prompt;
separation = mean(logodds | probe) - mean(logodds | no probe). Exact binomial/permutation null for acc.
Usage: python vera_step2.py file.json [file.json ...]
"""
import sys, json
import numpy as np

rng = np.random.default_rng(0)
for f in sys.argv[1:]:
    j = json.load(open(f))
    R = j["rows"] if "rows" in j else j["test_rows"]
    if "acc" not in j and "test" in j:
        j["acc"] = j["test"]["acc"]
    Y, N = j["yes_id"], j["no_id"]
    probe = np.array([r["probe"] for r in R]); top = np.array([r["top"] for r in R])
    lo = np.array([r["logodds_yes_no"] for r in R])
    corr = np.where(probe == 1, top == Y, top == N)
    pids = [r["prompt"] for r in R]
    byp = {}
    for p, c, pr in zip(pids, corr, probe):
        byp.setdefault(p, {})[pr] = c
    complete = [v for v in byp.values() if len(v) == 2]
    both = np.mean([v[0] and v[1] for v in complete])
    sep = lo[probe == 1].mean() - lo[probe == 0].mean()
    # paired 'sign' accuracy: does logodds(probe) > logodds(no probe) within each prompt (threshold-free)
    lop = {}
    for p, l, pr in zip(pids, lo, probe):
        lop.setdefault(p, {})[pr] = l
    gt = np.mean([v[1] > v[0] for v in lop.values() if len(v) == 2])
    eq = np.mean([v[1] == v[0] for v in lop.values() if len(v) == 2])
    # label-shuffle null on the SAME outputs: permute probe labels within prompt pairs (swap with p=0.5)
    accs = []
    for _ in range(2000):
        sw = {p: rng.random() < 0.5 for p in byp}
        pr2 = np.array([1 - pr if sw[p] else pr for p, pr in zip(pids, probe)])
        accs.append(np.mean(np.where(pr2 == 1, top == Y, top == N)))
    accs = np.array(accs)
    other = np.mean((top != Y) & (top != N))
    print(f"{f.split('/')[-1]}: n={len(R)} prompts={len(byp)} acc={corr.mean():.4f} both={both:.4f} sep={sep:.3f} "
          f"pair logodds probe>none {gt:.3f} (ties {eq:.3f}) non-Yes/No tops {other:.3f} | "
          f"within-pair label-swap null acc 95th pct {np.quantile(accs, .95):.4f}, max {accs.max():.4f}, "
          f"p={np.mean(accs >= corr.mean()):.4f} | json says acc={j.get('acc')}")
