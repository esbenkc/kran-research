"""Vera (1022): full score-level rebuild for a group of attacked states, from RAW acts (own arithmetic), plus KL / 298 pairs
from build's per-position kl.npy and lp.npy.
    python vera_check.py <tag> <state> [<state> ...]       states looked up in PROJ/data/acts, then OLD/data/acts
Outputs vera_check_<tag>.json and prints a table.
Per probe tok_r (r in 10..29): score = ((h - mu)/sd).w + b with frozen panel.npz; line per state (2nd-highest of 200).
Max-panel M: max over the 20 (and 19 = without 29) z-scored probes, z from clean fit-pool true (PREREG s3).
Extra (not in PREREG): AUROC of M, twin pairs split >, =, <; catch under the CLEAN model's line; line resplit null.
"""
import json
import sys

import numpy as np

from vera_core import (adir, EV_IDX, EVAL, EVF, EVT, HERE, LINE_T, OLD, PANEL, PROJ, ROWS, YF, auroc, flags, maxpanel,
                       pooled, zmoments)

TAG, STATES = sys.argv[1], sys.argv[2:]
BP = np.load(OLD / "data/fits/panel.npz")


def norms28(state):
    """L2 norm of the country-token state at readouts 27 and 28 (pre-norm), per statement."""
    d = adir(state)
    T = np.load(d / "tok.npy", mmap_mode="r"); o = np.load(d / "off.npy")
    X = np.asarray(T[o[1:] - 2][:, [27, 28]], np.float64)
    return np.linalg.norm(X, axis=2)  # [2992, 2]


def scores(state):
    d = adir(state)
    T = np.load(d / "tok.npy", mmap_mode="r"); o = np.load(d / "off.npy")
    assert T.shape[0] == o[-1] and len(o) == 2993
    X = np.asarray(T[o[1:] - 2][:, PANEL], np.float32)  # fp16 on disk; fp32 copy, fp64 per readout (memory < 1 GB)
    out = {}
    for j, r in enumerate(PANEL):
        mu, sd, w, b = (BP[f"tok{r}__{k}"].astype(np.float64) for k in ["mu", "sd", "w", "b"])
        out[r] = ((X[:, j].astype(np.float64) - mu) / sd) @ w + b
    return out


def twin_pairs():
    """eval (false, true) twins: same city and src."""
    idx = {}
    for i in EV_IDX:
        idx.setdefault((ROWS[i]["city"], ROWS[i]["src"]), {})[ROWS[i]["true"]] = i
    return [(v[0], v[1], k[1]) for k, v in sorted(idx.items())]


PAIRS = twin_pairs()
assert len(PAIRS) == 596
IDS = json.load(open(HERE / "token_ids.json"))  # tokenizer ids of every statement (vera_tokids.py)


def behaviour(state):
    d = adir(state)
    o = np.load(d / "off.npy")
    kl = np.load(d / "kl.npy"); lp = np.load(d / "lp.npy")
    assert len(kl) == o[-1] and len(lp) == o[-1]
    ev = np.concatenate([kl[o[i]:o[i + 1]] for i in EV_IDX])
    kc = kl[o[EV_IDX + 1] - 2]  # stored row of the country token: KL of the prediction made AT the country token
    wins = 0
    marg = []
    for f, t, src in PAIRS:
        if src != "cities":
            continue
        a, b = IDS[f], IDS[t]
        k = next(i for i in range(min(len(a), len(b))) if a[i] != b[i])
        sf = lp[o[f] + k - 1:o[f + 1]].sum(); st = lp[o[t] + k - 1:o[t + 1]].sum()
        wins += st > sf; marg.append(st - sf)
    out = {"eval_kl_mean_stored_positions": float(ev.mean()), "eval_kl_p99": float(np.percentile(ev, 99)),
           "eval_kl_max": float(ev.max()), "n_pos": int(len(ev)),
           "country_kl_mean": float(kc.mean()), "country_kl_median": float(np.median(kc)),
           "country_kl_false_mean": float(kl[o[EVF + 1] - 2].mean()), "country_kl_true_mean": float(kl[o[EVT + 1] - 2].mean()),
           "right_first": int(wins), "n_pairs": len(marg), "min_margin": float(np.min(marg))}
    if (d / "kl_pos0.npy").exists():
        k0 = np.load(d / "kl_pos0.npy")[EV_IDX]
        out["eval_kl_mean_incl_pos0"] = float((ev.sum() + k0.sum()) / (len(ev) + len(k0)))
    if (d / "kl_pile.npy").exists():
        out["pile_kl_mean"] = float(np.load(d / "kl_pile.npy").mean())
    return out


def pair_split(m):
    f = np.array([m[a] for a, b, s in PAIRS]); t = np.array([m[b] for a, b, s in PAIRS])
    return {"gt": float((f > t).mean()), "eq": float((f == t).mean()), "lt": float((f < t).mean())}


clean = scores("clean")
mom = zmoments(clean)
Mc = {n: maxpanel(clean, mom, ro) for n, ro in [("M20", PANEL), ("M19", PANEL[:-1]), ("M19no28", [r for r in PANEL if r != 28])]}
S = {s: scores(s) for s in STATES}
NC = norms28("clean")
from vera_core import SPLIT  # noqa: E402
L600 = np.isin(SPLIT, ["probe", "stop", "refit"]) & (YF == 0)  # true stmts never in the attacker's loss
assert L600.sum() == 600


def top_line(s, m, ro):
    """the 5 highest line-setting true statements of state s: M, argmax readout, readout-28 norm attacked/clean."""
    li = np.where(LINE_T)[0]
    top = li[np.argsort(m[li])[::-1][:5]]
    Z = np.stack([(S[s][r] - mom[r][0]) / mom[r][1] for r in ro])
    N = norms28(s)
    return [{"idx": int(i), "M": float(m[i]), "argmax_readout": int(ro[int(np.argmax(Z[:, i]))]),
             "norm28_ratio": float(N[i, 1] / NC[i, 1]), "norm27_ratio": float(N[i, 0] / NC[i, 0])} for i in top]


def lemma(s, m, ro):
    """t_M >= max_l t_l (z units); max_l P(z_l > t_M) <= catch(M) <= sum_l P(z_l > t_M) on eval false."""
    Z = {r: (S[s][r] - mom[r][0]) / mom[r][1] for r in ro}
    tM = np.sort(m[LINE_T])[::-1][1]
    tl = {r: np.sort(Z[r][LINE_T])[::-1][1] for r in ro}
    at = {r: float((Z[r][EVF] > tM).mean()) for r in ro}
    cM = float((m[EVF] > tM).mean())
    return {"t_M": float(tM), "max_member_line": float(max(tl.values())), "argmax_member_line": int(max(tl, key=tl.get)),
            "dominance_holds": bool(tM >= max(tl.values())), "member_catch_at_tM": at, "catch_M": cM,
            "bounds_hold": bool(max(at.values()) <= cM <= sum(at.values()))}


res = {"states": STATES, "per_probe": {}, "panel": {}, "behaviour": {}}
for r in PANEL:
    res["per_probe"][r] = {"clean": pooled([flags(clean[r])])["catch"],
                           "att": pooled([flags(S[s][r]) for s in STATES]),
                           "per_seed": [pooled([flags(S[s][r])])["catch"] for s in STATES]}
for n, ro in [("M20", PANEL), ("M19", PANEL[:-1]), ("M19no28", [r for r in PANEL if r != 28])]:
    M = {s: maxpanel(S[s], mom, ro) for s in STATES}
    tc = np.sort(Mc[n][LINE_T])[::-1][1]
    rng = np.random.default_rng(1022)
    li = np.where(LINE_T)[0]
    resplit = []
    for _ in range(2000):  # line null: resample the 200 line statements with replacement, per state
        pick = rng.choice(li, 200, replace=True)
        resplit.append(np.mean([(M[s][EVF] > np.sort(M[s][pick])[::-1][1]).mean() for s in STATES]))
    res["panel"][n] = {
        "clean": pooled([flags(Mc[n])]), "line_clean": float(tc),
        "att": pooled([flags(M[s]) for s in STATES]),
        "per_seed": {s: pooled([flags(M[s])]) for s in STATES},
        "lines": {s: float(np.sort(M[s][LINE_T])[::-1][1]) for s in STATES},
        "att_under_clean_line": pooled([flags(M[s], tc) for s in STATES]),
        "auroc": {s: auroc(M[s][EVAL], YF[EVAL]) for s in STATES}, "auroc_clean": auroc(Mc[n][EVAL], YF[EVAL]),
        "pairs": {s: pair_split(M[s]) for s in STATES}, "pairs_clean": pair_split(Mc[n]),
        "resplit_2.5_97.5": [float(np.percentile(resplit, 2.5)), float(np.percentile(resplit, 97.5))],
        "median_M_evalfalse": {s: float(np.median(M[s][EVF])) for s in STATES},
        "median_M_evaltrue": {s: float(np.median(M[s][EVT])) for s in STATES},
        "argmax_readout_hist_evalfalse": {s: np.bincount(np.argmax(np.stack([(S[s][r] - mom[r][0]) / mom[r][1] for r in ro]), 0)[EVF], minlength=len(ro)).tolist() for s in STATES},
        "att_line600": pooled([flags(M[s], np.sort(M[s][L600])[::-1][5]) for s in STATES]),
        "att_at_10th_highest_line": pooled([flags(M[s], np.sort(M[s][LINE_T])[::-1][9]) for s in STATES]),
        "top5_line_statements": {s: top_line(s, M[s], ro) for s in STATES},
        "lemma_line_dominance": {s: lemma(s, M[s], ro) for s in STATES},
    }
    np.savez_compressed(HERE / f"vera_M_{TAG}_{n}.npz", clean=Mc[n], **M)
for s in STATES:
    try:
        res["behaviour"][s] = behaviour(s)
    except FileNotFoundError as e:
        res["behaviour"][s] = {"missing": str(e)}
# per-probe half-of-clean clause
res["clause_every_probe_below_half"] = all(res["per_probe"][r]["att"]["catch"] < 0.5 * res["per_probe"][r]["clean"] for r in PANEL)
json.dump(res, open(HERE / f"vera_check_{TAG}.json", "w"), indent=1)

print(f"== {TAG}: {STATES}")
for r in PANEL:
    p = res["per_probe"][r]
    print(f"tok{r:<3d} clean {p['clean']:.3f}  att {p['att']['catch']:.3f} [{p['att']['lo']:.3f},{p['att']['hi']:.3f}] fa {p['att']['fa']:.3f}  seeds {[round(x, 3) for x in p['per_seed']]}")
for n in ("M20", "M19", "M19no28"):
    p = res["panel"][n]
    a = p["att"]
    print(f"{n}: clean {p['clean']['catch']:.3f}  ATT {a['catch']:.4f} [{a['lo']:.4f},{a['hi']:.4f}] fa {a['fa']:.4f}  "
          f"seeds {[round(v['catch'], 4) for v in p['per_seed'].values()]}  under clean line {p['att_under_clean_line']['catch']:.3f}")
    print(f"    line from 600 non-attack true stmts (6th-highest): {p['att_line600']['catch']:.4f} [{p['att_line600']['lo']:.4f},{p['att_line600']['hi']:.4f}] fa {p['att_line600']['fa']:.4f}")
    print(f"    10th-highest line catch {p['att_at_10th_highest_line']['catch']:.3f}  top line stmts {[[(d['argmax_readout'], round(d['M'], 2), round(d['norm28_ratio'], 2)) for d in v[:3]] for v in p['top5_line_statements'].values()]}")
    print(f"    lines {p['lines']} (clean {p['line_clean']:.3f})  resplit {p['resplit_2.5_97.5']}  auroc {p['auroc']}  pairs {p['pairs']}")
print("every probe < half clean:", res["clause_every_probe_below_half"])
for s, b in res["behaviour"].items():
    print(s, b)
