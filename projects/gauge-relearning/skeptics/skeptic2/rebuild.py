"""skeptic2: independent rebuild of steps-to-recover from raw runs.jsonl + null tests.

Usage: uv run --project kran-research python rebuild.py <runs.jsonl> [threshold]
Does not import any build/plan code. Recomputes everything from per-eval rows.
"""
import json
import math
import sys
from collections import defaultdict
from itertools import permutations
import random

import numpy as np
from scipy import stats


def load(path, metric="acc_A", model="unlearned"):
    """Reads raw per-step rows: a flat .jsonl file, or every *.jsonl under a dir (seed*/...).
    Fields from the row when present, else parsed from '<att>_log2c<K>_lr<LR>.jsonl'."""
    import pathlib
    import re

    p = pathlib.Path(path)
    files = [p] if p.is_file() else sorted(f for f in p.rglob("*.jsonl") if not f.name.startswith("devcpu_"))
    rows = []
    for f in files:
        m = re.match(r"(.+?)_log2c(-?\d+)_lr(.+)\.jsonl", f.name)
        sm = re.search(r"seed(\d+)", str(f.parent))
        for l in open(f):
            if not l.strip():
                continue
            r = json.loads(l)
            if "step" not in r or metric not in r:
                continue
            if r.get("model", "unlearned") != model:
                continue
            att = r["attacker"] if "attacker" in r else m.group(1)
            if "c" in r:
                c = r["c"]
            else:
                c = 2.0 ** int(r["log2c"] if "log2c" in r else m.group(2))
            lr = r["lr"] if "lr" in r else float(m.group(3))
            seed = r["seed"] if "seed" in r else (int(sm.group(1)) if sm else 0)
            rows.append(dict(attacker=att, c=float(c), seed=int(seed), lr=float(lr), step=int(r["step"]),
                             acc_B=r[metric], acc_heldout=r.get("acc_B"), order=int(r.get("order", 0)),
                             loss=r.get("loss", r.get("forget_train_loss"))))
    return rows

def paired_ratios(rows, thr, cap):
    """#472 rule: ratio steps(c)/steps(c=1) within the same (attacker, seed, order) at the tuned lr."""
    runs = defaultdict(list)
    for r in rows:
        runs[(r["attacker"], r["c"], r["lr"], r["seed"], r["order"])].append(r)
    hit = {k: hitting(v, thr, cap) for k, v in runs.items()}
    # tuned lr per (attacker,c) = the lr the extra-order runs used; else seed-0 order-0 fastest
    tuned = {}
    for (a, c, lr, s, o) in hit:
        if o > 0:
            tuned[(a, c)] = lr
    for (a, c, lr, s, o), st in hit.items():
        if (a, c) not in tuned and s == 0 and o == 0:
            cands = [(st2[1], lr2) for (a2, c2, lr2, s2, o2), st2 in hit.items()
                     if a2 == a and c2 == c and s2 == 0 and o2 == 0 and st2[0] == "recovered"]
            if cands:
                tuned[(a, c)] = min(cands)[1]
    cell = {}
    for (a, c, lr, s, o), st in hit.items():
        if tuned.get((a, c)) == lr:
            cell[(a, c, s, o)] = st
    print("PAIRED RATIOS (#472 rule: median>=1.5 AND >1 in >=80% of pairs), all seeds x orders at tuned lr")
    for a in sorted({k[0] for k in cell}):
        for c in sorted({k[1] for k in cell if k[0] == a}):
            if c == 1.0:
                continue
            rs = []
            for (a2, c2, s, o), st in cell.items():
                if a2 == a and c2 == c and (a, 1.0, s, o) in cell:
                    b = cell[(a, 1.0, s, o)]
                    if st[0] != "diverged" and b[0] != "diverged":
                        rs.append((s, o, st[1] / max(b[1], 1)))
            if not rs:
                continue
            v = sorted(x[2] for x in rs)
            med = float(np.median(v))
            frac = sum(x > 1 for x in v) / len(v)
            verdict = "PASS" if med >= 1.5 and frac >= 0.8 else "fail"
            print(f"  {a:8s} log2c={int(math.log2(c)):+d} n={len(v)} median={med:.2f} >1:{frac:.0%} range={v[0]:.2f}-{v[-1]:.2f} {verdict}")
    print()


def key_of(r):
    return (r["attacker"], float(r["c"]), float(r["lr"]), int(r["seed"]))


def hitting(rows, thr, cap):
    """rows of one run -> (status, step)."""
    rows = sorted(rows, key=lambda r: r["step"])
    for r in rows:
        loss = r.get("loss")
        if loss is not None and not math.isfinite(float(loss)):
            return "diverged", None
        if r["acc_B"] >= thr:
            return "recovered", int(r["step"])
    return "censored", cap


def main():
    # usage: rebuild.py <raw dir|runs.jsonl> [metric=acc_A] [threshold=0.6 absolute] [cap=500] [model=unlearned]
    path = sys.argv[1]
    metric = sys.argv[2] if len(sys.argv) > 2 else "acc_A"
    thr_abs = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9
    cap = int(sys.argv[4]) if len(sys.argv) > 4 else 500
    model = sys.argv[5] if len(sys.argv) > 5 else "unlearned"
    all_rows = load(path, metric, model)
    paired_ratios(all_rows, thr_abs, cap)
    rows = [r for r in all_rows if r["order"] == 0]
    runs = defaultdict(list)
    for r in rows:
        runs[key_of(r)].append(r)
    thr = defaultdict(lambda: thr_abs)
    res = {k: hitting(v, thr[k[3]], cap) for k, v in runs.items()}
    print(f"metric={metric} threshold>={thr_abs} cap={cap} model={model} runs={len(res)}")
    # held-out acc_B at the end of each best run is reported below
    final_heldout = {k: max(v, key=lambda r: r["step"]).get("acc_heldout") for k, v in runs.items()}
    # report step-0 acc_B per run (should be identical across c within attacker/seed: gauge preserves function)
    s0 = defaultdict(set)
    for k, v in runs.items():
        r0 = min(v, key=lambda r: r["step"])
        s0[(k[3])].add(round(r0["acc_B"], 6))
    print("step-0 acc_B values per seed (expect 1 value each):", {k: sorted(v) for k, v in s0.items()})

    attackers = sorted({k[0] for k in res})
    cs = sorted({k[1] for k in res})
    seeds = sorted({k[3] for k in res})
    # best lr per (attacker,c,seed): min step among recovered, tie -> lower lr; else censored at lowest lr
    def best(a, c, s):
        cands = sorted((k[2], res[k]) for k in res if k[0] == a and k[1] == c and k[3] == s)
        if not cands:
            return None, None
        ok = [(st[1], lr) for lr, st in cands if st[0] == "recovered"]
        if ok:
            lr = min(ok)[1]
        else:
            cens = [lr for lr, st in cands if st[0] == "censored"]
            lr = cens[0] if cens else cands[0][0]
        return lr, res[(a, c, lr, s)]

    print("\nTABLE steps-to-recover at best lr per seed; status R/C/D")
    table = {}
    for a in attackers:
        print(f"\n{a}")
        for c in cs:
            per = [(s,) + best(a, c, s) for s in seeds]
            per = [(s, lr, st) for s, lr, st in per if st]
            table[(a, c)] = [st for _, _, st in per]

            def edge(s, lr):
                g = sorted(k[2] for k in res if k[0] == a and k[1] == c and k[3] == s)
                return "*" if len(g) > 1 and lr in (g[0], g[-1]) else ""

            fmt = ", ".join(
                f"{st[1]}{st[0][0].upper()}@{lr:g}{edge(s, lr)} hB={final_heldout[(a, c, lr, s)]}"
                for s, lr, st in per
            )
            steps = [st[1] for _, _, st in per if st[0] != "diverged"]
            med = np.median(steps) if steps else float("nan")
            grid = sorted((k[2], res[k]) for k in res if k[0] == a and k[1] == c and k[3] == seeds[0])
            gfmt = " ".join(f"{lr:g}:{st[1]}{st[0][0].upper()}" for lr, st in grid)
            print(f"  log2c={math.log2(c):+.0f} median={med} seeds=[{fmt}]\n      grid(seed{seeds[0]})= {gfmt}")

    print("\nNULL / TREND TESTS per attacker (runs at best lr, censored at cap, diverged dropped)")
    for a in attackers:
        xs, ys = [], []
        for c in cs:
            for st in table[(a, c)]:
                if st[0] != "diverged":
                    xs.append(math.log2(c))
                    ys.append(math.log(st[1] + 1))
        if len(set(xs)) < 2:
            continue
        rho, _ = stats.spearmanr(xs, ys)
        rho_abs, _ = stats.spearmanr([abs(x) for x in xs], ys)

        def kw(xv):
            groups = defaultdict(list)
            for x, y in zip(xv, ys):
                groups[x].append(y)
            g = [v for v in groups.values()]
            if any(len(v) < 1 for v in g) or len(g) < 2:
                return 0.0
            try:
                return stats.kruskal(*g).statistic
            except ValueError:
                return 0.0

        # permutation null: c labels carry no information (shuffle across runs)
        rng = random.Random(0)
        N = 5000
        h_obs = kw(xs)
        n_ge = n_ge_abs = 0
        for _ in range(N):
            xp = xs[:]
            rng.shuffle(xp)
            if kw(xp) >= h_obs - 1e-12:
                n_ge += 1
            r, _ = stats.spearmanr([abs(x) for x in xp], ys)
            if r >= rho_abs - 1e-12:
                n_ge_abs += 1
        rho_s = f"rho(log2c)={rho:+.2f} rho(|log2c|)={rho_abs:+.2f} perm-p(|log2c|, 1-sided)={n_ge_abs / N:.3f} KW-perm-p={n_ge / N:.3f}"

        def med(c):
            v = [st[1] for st in table[(a, c)] if st[0] != "diverged"]
            return float(np.median(v)) if v else float("nan")

        c1 = [st[1] for st in table.get((a, 1.0), []) if st[0] != "diverged"]
        spread = (max(c1) / max(min(c1), 1)) if c1 else float("nan")
        meds = {c: med(c) for c in cs}
        ok = [v for v in meds.values() if not math.isnan(v)]
        print(
            f"  {a:12s} {rho_s}\n"
            f"  {'':12s} median max/min across c={max(ok) / max(min(ok), 1):.2f}x  c=1 seed max/min={spread:.2f}x  "
            f"medians by log2c={ {int(math.log2(c)): meds[c] for c in cs} }"
        )

    print("\nDEFENDER VIEW: per c, fastest attacker (median steps at best lr). Gauge 'survives' only if this exceeds c=1's value beyond seed spread.")
    for c in cs:
        best = []
        for a in attackers:
            v = [st[1] for st in table[(a, c)] if st[0] != "diverged"]
            if v:
                best.append((float(np.median(v)), a))
        if best:
            m = min(best)
            print(f"  log2c={int(math.log2(c)):+d} fastest={m[1]} median={m[0]}")

    print("\nNULL: renorm maps every c to the same weights -> trajectories at same (lr,seed) should match across c.")
    if "renorm" in attackers:
        for s in seeds:
            for lr in sorted({k[2] for k in res if k[0] == "renorm" and k[3] == s}):
                traj = {}
                for c in cs:
                    k = ("renorm", c, lr, s)
                    if k in runs:
                        traj[c] = [r["acc_B"] for r in sorted(runs[k], key=lambda r: r["step"])]
                if len(traj) < 2:
                    continue
                ref = traj.get(1.0, next(iter(traj.values())))
                mism = {int(math.log2(c)): sum(x != y for x, y in zip(t, ref)) + abs(len(t) - len(ref)) for c, t in traj.items()}
                hits = {int(math.log2(c)): res[("renorm", c, lr, s)][1] for c in traj}
                print(f"  seed{s} lr={lr:g} steps-to-recover by log2c={hits} acc-mismatch-count vs c=1={mism}")


if __name__ == "__main__":
    main()
