"""Score every fine-tune run of attack_proj.py (and the theory-1023 post-norm runs, data/runs1023/) by the frozen rules
of PREREG.md (7c8f379) + DEVIATIONS.md D1/D2.

Per run and checkpoint (steps 0, 50, ..., 300):
    catch          own probe, re-set line = 2nd-highest score of the 200 refit TRUE statements of THAT state, share of the
                   596 eval FALSE statements above it (1019-1023 rule); also at the probe's CLEAN line
    kl_country     mean over the 1192 eval statements of KL(p_clean || p_attacked) at the country-token position
                   (the next-token distribution right after the country name) = the title metric's unit
    kl_all         mean over every position of the eval statements (the prereg's 'eval KL', defender view)
    right_first    298-pair right-country ranking (metrics.ranking), at every checkpoint
    rescore        catch of every other probe in probes.npz on the same attacked states (e.g. lit_k* = P_F w)
Per arm (pooled over seeds):
    pooled catch(step) = mean of per-seed catch; seed-mean kl_country(step)
    PRICE (D1)     = smallest seed-mean kl_country over checkpoints 50..300 with pooled catch <= 0.20 (inf if none);
                     per seed: same on its own curve; first crossing (linear interpolation from step 0) reported too
STATUS (PREREG s5 + Tove #6034): YES = price(P) >= 3 price(U) AND R falls (pooled <= 20%, price(R) within 1.5x of U) AND
min_seed price(P) > max_seed price(U); OR P never <= 20% with step-300 pooled catch >= 50% AND R falls.
NO = price(P) <= 1.5 price(U) AND max_seed price(P) <= 1.5 max_seed price(U). Else UNCONFIRMED.
Outputs: data/results/results.json, data/results/summary_<run>.json (one per run, for the writer and Figure 1).
Usage: PY analysis.py [--u-source new|1023|both]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from common import country_pos, load_split, rows, threshold, tokenize
from metrics import boot_catch, ranking

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "data" / "runs"
R1023 = ROOT / "data" / "runs1023"
OUT = ROOT / "data" / "results"
STEPS = list(range(0, 301, 50))
_, Y, CITY, SPLIT = rows()
EV = np.where(SPLIT == "eval")[0]
EVF = np.where((SPLIT == "eval") & (Y == 1))[0]
LINE = np.where((SPLIT == "refit") & (Y == 0))[0]


def load_ckpts(d):
    out = {}
    for st in STEPS:
        f = d / f"ckpt_{st}.npz"
        if f.exists():
            out[st] = dict(np.load(f, allow_pickle=False))
    return out


def own_column(ck, probe):
    if "probes" in ck:
        names = [str(x) for x in ck["probes"]]
        return names.index(probe), names
    return list(ck["scored"]).index(29), ["tok27", "tok28", "tok29"]       # theory-1023 files: tok29 = unproj


def score_run(d, probe, cpos):
    cks = load_ckpts(d)
    j, names = own_column(cks[0], probe)
    s0 = cks[0]["scores"][:, j]
    line_clean = threshold(s0[LINE])
    rows_ = []
    for st, ck in sorted(cks.items()):
        s = ck["scores"][:, j]
        kl, koff = ck["kl"], ck["koff"]
        line = threshold(s[LINE])
        r = {"step": st, "catch": float((s[EVF] > line).mean()), "catch_clean_line": float((s[EVF] > line_clean).mean()),
             "line": line, "kl_country": float(np.mean([kl[koff[i] + cpos[i]] for i in EV])),
             "kl_all": float(np.nanmean(np.concatenate([kl[koff[i]:koff[i + 1]] for i in EV]))),
             "false_alarm_eval": float((s[np.where((SPLIT == "eval") & (Y == 0))[0]] > line).mean())}
        if "lp" in ck:
            rk = ranking(ck["lp"], koff)
            r["right_first"] = rk["right_first"]
        if "probes" in ck:
            r["rescore"] = {n: float((ck["scores"][EVF, a] > threshold(ck["scores"][LINE, a])).mean())
                            for a, n in enumerate(names) if n != probe}
        rows_.append(r)
    return rows_, s0, cks


def price_of(curve_steps, catch, kl):
    """D1: smallest kl over checkpoints 50..300 with catch <= 0.20 (inf if none); first crossing interpolated."""
    ok = [k for st, c, k in zip(curve_steps, catch, kl) if st >= 50 and c <= 0.20]
    pr = min(ok) if ok else float("inf")
    fc = float("inf")
    for a in range(1, len(curve_steps)):
        if catch[a] <= 0.20 < catch[a - 1]:
            t = (catch[a - 1] - 0.20) / (catch[a - 1] - catch[a])
            fc = kl[a - 1] + t * (kl[a] - kl[a - 1])
            break
    return pr, fc


def arm_summary(runs):
    """runs: list of (seed, rows). Pools only checkpoints present in every seed."""
    common = sorted(set.intersection(*[{r["step"] for r in rr} for _, rr in runs]))
    by = {s: {r["step"]: r for r in rr} for s, rr in runs}
    pc = [float(np.mean([by[s][st]["catch"] for s, _ in runs])) for st in common]
    kc = [float(np.mean([by[s][st]["kl_country"] for s, _ in runs])) for st in common]
    ka = [float(np.mean([by[s][st]["kl_all"] for s, _ in runs])) for st in common]
    price, first = price_of(common, pc, kc)
    per_seed = {}
    for s, rr in runs:
        st = [r["step"] for r in rr]
        per_seed[s] = dict(zip(("price", "first_cross"), price_of(st, [r["catch"] for r in rr], [r["kl_country"] for r in rr])))
        per_seed[s]["final"] = {k: rr[-1][k] for k in ("step", "catch", "kl_country", "kl_all", "right_first") if k in rr[-1]}
    last = common[-1]
    conv = None
    if len(common) >= 2 and common[-2] >= 0.8 * last:
        conv = {"steps": [common[-2], last], "pooled_catch": [pc[-2], pc[-1]], "kl_country": [kc[-2], kc[-1]]}
    return {"seeds": [s for s, _ in runs], "steps": common, "pooled_catch": pc, "kl_country": kc, "kl_all": ka,
            "price": price, "first_cross": first, "per_seed": per_seed, "final_step": last,
            "final_pooled_catch": pc[-1], "convergence_last20pct": conv,
            "right_first_min_final": min(by[s][last].get("right_first", 999) for s, _ in runs)}


def status(U, P, R):
    def falls(A):
        return A["price"] < float("inf")
    pu, pp, pr = U["price"], P["price"], R["price"]
    seeds_p = [v["price"] for v in P["per_seed"].values()]
    seeds_u = [v["price"] for v in U["per_seed"].values()]
    r_falls = falls(R) and pr <= 1.5 * pu
    yes_price = falls(P) and pp >= 3 * pu and r_falls and min(seeds_p) > max(seeds_u)
    yes_surv = (not falls(P)) and P["final_pooled_catch"] >= 0.50 and r_falls
    no = falls(P) and pp <= 1.5 * pu and max(seeds_p) <= 1.5 * max(seeds_u)
    enough = len(P["seeds"]) >= 2 and len(R["seeds"]) >= 2          # PREREG s7: fewer than 2 seeds per arm = UNCONFIRMED
    st = ("YES" if (yes_price or yes_surv) else ("NO" if no else "UNCONFIRMED")) if enough else "UNCONFIRMED"
    return {"status": st, "enough_seeds": enough, "ratio_P_U": pp / pu if pu > 0 else None, "ratio_R_U": pr / pu if pu > 0 else None,
            "r_falls_within_1.5x": r_falls, "yes_price_branch": yes_price, "yes_survival_branch": yes_surv,
            "no_branch": no, "seed_prices_P": seeds_p, "seed_prices_U": seeds_u}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--u-source", default="new", choices=["new", "1023", "both"])
    ap.add_argument("--horizon", type=int, default=None, help="D8: read prices over checkpoints <= horizon (default: data/HORIZON or 300)")
    a = ap.parse_args()
    hz = a.horizon or (int((ROOT / "data" / "HORIZON").read_text().strip()) if (ROOT / "data" / "HORIZON").exists() else 300)
    global STEPS
    STEPS = [s for s in range(0, 301, 50) if s <= hz]
    from transformers import AutoTokenizer
    from common import MODEL
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tokenize(tok, [r["text"] for r in load_split()])
    cpos, _ = country_pos(tok, ids)
    OUT.mkdir(parents=True, exist_ok=True)
    arms = {"unproj": [], "proj": [], "rand": []}
    finals = {"unproj": [], "proj": [], "rand": []}
    per_run = {}
    for d in sorted(RUNS.glob("*__lam10000__seed*")):
        probe, _, seed = d.name.split("__")
        if probe not in arms or not (d / "ckpt_0.npz").exists() or int(seed[4:]) >= 90:
            continue
        rr, s0, cks = score_run(d, probe, cpos)
        if probe == "unproj" and max(r["step"] for r in rr) < 150:
            # D8: the 50-step reproduction of the theory-1023 U seed-0 run with the --gc code
            ref, _, rck = score_run(R1023 / f"r29__lam10000__seed{int(seed[4:])}", "unproj", cpos)
            mine = {r["step"]: r for r in rr}
            theirs = {r["step"]: r for r in ref}
            st = max(k for k in mine if k in theirs)
            j_m, _ = own_column(cks[st], "unproj")
            j_t, _ = own_column(rck[st], "unproj")
            sm, stt = cks[st]["scores"][EV, j_m], rck[st]["scores"][EV, j_t]
            res_repro = {"run": d.name, "step": st, "new": mine[st], "theory1023": theirs[st],
                         "kl_country_ratio": mine[st]["kl_country"] / theirs[st]["kl_country"],
                         "eval_score_corr": float(np.corrcoef(sm, stt)[0, 1]),
                         "eval_score_mean_abs_diff": float(np.mean(np.abs(sm - stt))),
                         "rule": "D8: within 20% of the 1023 s0 run at step 50 (country-token KL; catch reported)"}
            res_repro["pass_20pct"] = abs(res_repro["kl_country_ratio"] - 1) <= 0.20
            (OUT / f"repro_{d.name}.json").write_text(json.dumps(res_repro, indent=1))
            continue
        if a.u_source == "1023" and probe == "unproj":
            continue
        arms[probe].append((f"new{seed[4:]}", rr))
        per_run[d.name] = rr
        (OUT / f"summary_{d.name}.json").write_text(json.dumps({"run": d.name, "probe": probe, "checkpoints": rr}, indent=1))
    ctx1023 = []
    for d in sorted(R1023.glob("r29__lam10000__seed*")):
        rr, _, _ = score_run(d, "unproj", cpos)
        per_run[f"1023_{d.name}"] = rr
        ctx1023.append((f"1023s{d.name[-1]}", rr))
    res = {"rules": "PREREG 7c8f379 + DEVIATIONS D1-D10", "u_source": a.u_source, "horizon": hz, "arms": {}, "per_run": per_run}
    if a.u_source == "1023":
        arms["unproj"] = ctx1023
    elif a.u_source == "both":
        arms["unproj"] = arms["unproj"] + ctx1023
    else:
        # D10 (Felix #6204, Tove #6208): U arm = new-code seeds; the 1023 runs are context. If only ONE new-code U seed
        # exists, the 1023 seeds join only when new U s0's price is within 20% of 1023 s0's price (T2 criterion).
        res["arm_unproj_1023_context"] = arm_summary(ctx1023)
        new_u = arms["unproj"]
        if new_u and any(t == "new0" for t, _ in new_u):
            pn0 = arm_summary([x for x in new_u if x[0] == "new0"])["per_seed"]["new0"]["price"]
            p230 = res["arm_unproj_1023_context"]["per_seed"]["1023s0"]["price"]
            res["d10_u0_vs_1023s0"] = {"new_u_s0_price": pn0, "u1023_s0_price": p230, "ratio": pn0 / p230}
        chain_log = ROOT / "data" / "logs" / "chain.log"
        slipped = chain_log.exists() and "SKIP unproj__lam10000__seed1" in chain_log.read_text()
        if len(new_u) == 1 and slipped:     # slip case only: U s1 could not start by 18:20 (chain.log SKIP line)
            pn = arm_summary(new_u)["per_seed"][new_u[0][0]]["price"]
            p23 = arm_summary(ctx1023)["per_seed"]["1023s0"]["price"]
            ok = abs(pn / p23 - 1) <= 0.20 if np.isfinite(pn) else False
            res["d10_slip_check"] = {"new_u_s0_price": pn, "u1023_s0_price": p23, "within_20pct": ok, "slipped": True}
            if ok:
                arms["unproj"] = new_u + [x for x in ctx1023 if x[0] != "1023s0"]
    for k, runs in arms.items():
        if runs:
            res["arms"][k] = arm_summary(runs)
    if all(res["arms"].get(k) for k in ("unproj", "proj", "rand")):
        res["status"] = status(res["arms"]["unproj"], res["arms"]["proj"], res["arms"]["rand"])
        if len(res["arms"]["unproj"]["seeds"]) < 2:
            res["status"]["status"] = "UNCONFIRMED"
            res["status"]["note"] = "U has fewer than 2 seeds (D10)"
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=lambda o: None if o == float("inf") else o))
    brief = {k: {kk: v[kk] for kk in ("seeds", "price", "first_cross", "final_pooled_catch", "right_first_min_final")}
             for k, v in res["arms"].items()}
    print(json.dumps({"arms": brief, "status": res.get("status")}, indent=1, default=str))


if __name__ == "__main__":
    main()
