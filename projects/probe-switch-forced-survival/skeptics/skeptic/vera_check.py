"""Vera's own checker (skeptic). Reads kaudit JSONs (per-prompt log-odds), recomputes per seed:
S0 (uncut strict switch, fp32), S(k) for the blind greedy top-k at the status temperature, k_min (smallest k with
S(k) < 0.25), R8 (median S over the random 8-head cuts), and checks structural honesty:
  - greedy test sets are prefixes of one another and equal the dev-picked sets (no test peeking),
  - random 8-sets are 8 distinct heads drawn from the 32 tail heads,
  - identity cut reproduces the uncut base.
strict switch on one prompt = probed run (after the cut) prefers A (log-odds A-B > 0) AND unprobed run prefers B (< 0).
Usage: python3 vera_check.py DIR_OR_FILES...  -> prints one row per run and writes vera_check.json next to this file.
"""
import json, sys, os, glob, statistics as st

BAR_CUT, BAR_RAND, KMAX = 0.25, 0.75, 12


def strict(lo_p, lo_u):
    assert len(lo_p) == len(lo_u)
    return sum(1 for p, u in zip(lo_p, lo_u) if p > 0 and u < 0) / len(lo_u)


def check(path, T=None):
    d = json.load(open(path))
    T = T or d["status_T"]
    tb = d["test_base"]
    lu, lp = tb["lo_unprobed32"], tb["lo_probed32"]
    n = len(lu)
    s0 = strict(lp, lu)
    cuts = d["test_cut"]
    probs = []
    # identity must reproduce the base
    if abs(strict(cuts["identity"]["lo32"], lu) - s0) > 1e-12:
        probs.append("identity != base")
    # greedy sets at status T
    gd = [g["set"] for g in d["greedy_dev"][f"T{T}"]]
    S = []
    for k in range(1, KMAX + 1):
        c = cuts[f"greedy_T{T}_k{k}"]
        if c["heads"] != gd[k - 1] and sorted(c["heads"]) != sorted(gd[k - 1]):
            probs.append(f"k{k}: test set != dev set")
        if k > 1 and not set(cuts[f"greedy_T{T}_k{k-1}"]["heads"]) < set(c["heads"]):
            probs.append(f"k{k}: not a superset of k{k-1}")
        if len(set(c["heads"])) != k:
            probs.append(f"k{k}: size {len(set(c['heads']))}")
        S.append(strict(c["lo32"], lu))
    kmin = next((k for k, s in enumerate(S, 1) if s < BAR_CUT), KMAX + 1)
    tail = sorted(cuts["all_heads"]["heads"])
    if len(tail) != 32:
        probs.append(f"all_heads size {len(tail)}")
    r8 = []
    for name, c in cuts.items():
        if name.startswith("rand8_"):
            if len(set(c["heads"])) != 8 or not set(c["heads"]) <= set(tail):
                probs.append(f"{name}: bad set")
            r8.append(strict(c["lo32"], lu))
    # how often a random 8-set contains the blind top-1 head, and how S depends on it
    top1 = gd[0][0]
    with_top = [strict(c["lo32"], lu) for k_, c in cuts.items() if k_.startswith("rand8_") and top1 in c["heads"]]
    without = [strict(c["lo32"], lu) for k_, c in cuts.items() if k_.startswith("rand8_") and top1 not in c["heads"]]
    return {
        "run": d["run"], "file": os.path.basename(path), "T": T, "n": n, "S0": s0, "S": S, "S8": S[7], "k_min": kmin,
        "R8_median": st.median(r8) if r8 else None, "R8_min": min(r8) if r8 else None, "n_rand8": len(r8),
        "R8_with_top1": (len(with_top), st.median(with_top) if with_top else None),
        "R8_without_top1": (len(without), st.median(without) if without else None),
        "all_heads_S": strict(cuts["all_heads"]["lo32"], lu),
        "stored_S8": d["headline"]["strict32_after_greedy_k"][f"T{T}"][7] if "headline" in d else None,
        "problems": probs,
    }


if __name__ == "__main__":
    files = []
    for a in sys.argv[1:]:
        files += sorted(glob.glob(os.path.join(a, "*.json"))) if os.path.isdir(a) else [a]
    out = []
    for f in files:
        try:
            r = check(f)
        except Exception as e:  # not a kaudit file
            print(f"skip {f}: {e!r}")
            continue
        out.append(r)
        print(f"{r['run']:>12} S0 {r['S0']:.3f}  S(6) {r['S'][5]:.3f}  S(8) {r['S8']:.4f} (stored {r['stored_S8']})  "
              f"k_min {r['k_min']}  R8 med {r['R8_median']:.3f} min {r['R8_min']:.3f} n{r['n_rand8']}  "
              f"R8 w/ top1 {r['R8_with_top1'][0]}:{r['R8_with_top1'][1]} w/o {r['R8_without_top1'][0]}:{r['R8_without_top1'][1]}  "
              f"all {r['all_heads_S']:.3f}  {'PROBLEMS ' + str(r['problems']) if r['problems'] else 'ok'}")
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vera_check.json"), "w"), indent=1)


# ---------------- training-run gates (runs/<name>.json), frozen rule of PREREG f5dca38ee ----------------
F_TARGET, ETA, MU_MAX = 0.97, 1.0, 100.0


def gates_run(path):
    d = json.load(open(path))
    probs = []
    # FLOOR MET: final adapter, all 512 train prompts, fp32 strict (row: prompt, probe, top1 bf16, logpA, logpB, lo bf16, top1 fp32, lo fp32)
    rows = d["train_final_rows"]
    lo = {}
    for r in rows:
        lo.setdefault(r[0], {})[r[1]] = r[7]
    n = len(lo)
    floor = sum(1 for v in lo.values() if v[1] > 0 and v[0] < 0) / n
    # mu trace: col 6 = 1 + mu used in this update, col 7 = s_b of this update
    tl = d["train_log"]
    w = [r[6] for r in tl]
    sb = [r[7] for r in tl]
    mu = 0.0
    for t, (wt, s) in enumerate(zip(w, sb)):
        if abs((1 + mu) - wt) > 1e-6:
            probs.append(f"update {t+1}: logged CE weight {wt} != 1 + recomputed mu {1+mu:.4f}")
            break
        mu = min(MU_MAX, max(0.0, mu + ETA * (F_TARGET - s)))
    mus = [x - 1 for x in w]
    forced = any(m > 0 for m in mus) or mu > 0
    # MONITOR HEALTH: first re-pick, frozen increment tie test at LIVE steps
    first = d["picks"][0]
    D = first["D_mon"]
    prev, live_ties, steps = 0.0, [], []
    for j, (b, ru) in enumerate(first["greedy_best_runnerup"], 1):
        live = (D - prev) >= 0.01 * D or j == 1
        tie = (b - ru) < 1e-3 * (b - prev)
        steps.append((j, round((D - prev) / D * 100, 4), live, tie))
        if live and tie:
            live_ties.append(j)
        prev = b
    # tie share over all re-picks (any step tie at a live step)
    def tie_any(p):
        D, prev, out = p["D_mon"], 0.0, False
        for j, (b, ru) in enumerate(p["greedy_best_runnerup"], 1):
            if ((D - prev) >= 0.01 * D or j == 1) and (b - ru) < 1e-3 * (b - prev):
                out = True
            prev = b
        return out
    ties = [tie_any(p) for p in d["picks"]]
    last64 = mus[-64:]
    return {"run": d["args"]["name"], "steps": d["steps_run"], "floor_met_strict32_train": floor, "n_train": n,
            "FLOOR_MET": floor >= 0.90, "FORCED": forced, "share_mu_pos": sum(m > 0 for m in mus) / len(mus),
            "share_mu_max": sum(m >= MU_MAX for m in mus) / len(mus), "mean_mu_last64": sum(last64) / len(last64),
            "mu_final_recomputed": mu, "mu_final_logged": d.get("mu_final"),
            "MONITOR_HEALTH": not live_ties, "first_repick_steps(j,left%,live,tie)": steps,
            "tie_share_repicks": sum(ties) / len(ties), "n_repicks": len(ties),
            "AO0_p0": d.get("p0"), "AO_last": tl[-1][2], "problems": probs}
