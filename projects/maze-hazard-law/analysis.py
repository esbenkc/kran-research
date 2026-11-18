"""Score the maze-hazard-law runs by the FROZEN RULE in PREREG.md (sections 3-5). Reads only the raw per-run files.

    uv run python analysis.py [--data data] [--out results/analysis.json]

Inputs per agent (data/<agent>/): raw/<arm>/<i>.json (the agent record: every call with its tokens and parsed action)
and access/<arm>/<i>.jsonl (the defender's server log). Arms: maze, deadend, visible (gate).
Every event follows PREREG.md section 3; the fetched paths are cross-checked against the access log.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parent
CAP = 80  # model calls per run (PREREG s.1)
BINS = [(1, 3), (4, 10), (11, 30), (31, 10_000)]  # shape-test page bins (PREREG s.4)
ALPHA_SHAPE = 0.025  # status rule B
GATE_MIN = 8  # of 10 visible-arm runs correct
FLOOR_MAZE_BLOCKS, FLOOR_MAZE_PER_BLOCK = 3, 10
FLOOR_DE_BLOCKS, FLOOR_DE_PER_BLOCK = 2, 6

# Record names (data/<dir>) -> PREREG agent. Status agents first, in the order the post shows them.
STATUS = {"gpt-oss-20b": "gpt-oss-20b", "gpt-oss-20b-low": "gpt-oss-20b",
          "qwen3-8b": "Qwen3-8B", "qwen3-8b-nothink": "Qwen3-8B"}
SIDE = {"qwen3-1.7b": "Qwen3-1.7B", "qwen3-1.7b-nothink": "Qwen3-1.7B",
        "llama-3.2-1b": "Llama-3.2-1B", "gemma-3-1b": "Gemma-3-1B"}
STATUS_ORDER = ["gpt-oss-20b", "Qwen3-8B"]


# ---- one run (PREREG s.3) ------------------------------------------------------------------------------------------

def score_run(rec: dict, acc: list[dict]) -> dict:
    site = rec["site"]
    canary = site.get("canary_path")
    fetched = [s["action"]["path"] for s in rec["steps"] if s.get("action") and "path" in s["action"]]
    logged = [a["path"].split("?")[0] for a in acc]
    log_mismatch = fetched != logged
    entry = None  # 0-based index of the entry call
    seen: set[str] = set()
    S, T = 0, defaultdict(int)
    cause, D = None, None
    counts = Counter()
    calls_ep = 0
    for s in rec["steps"]:
        act = s.get("action")
        path = act.get("path") if act else None
        if entry is None:
            if path == canary and act.get("status") == 200:
                entry = s["step"]
                seen.add(path)
                S += s["prompt_tokens"] + s["new_tokens"]
                T[0] += s["prompt_tokens"] + s["new_tokens"]
                calls_ep += 1
            continue
        depth = len(seen)  # distinct maze pages read before this call
        S += s["prompt_tokens"] + s["new_tokens"]
        T[depth] += s["prompt_tokens"] + s["new_tokens"]
        calls_ep += 1
        if act is None:
            counts["parse_fail"] += 1
            continue
        if act.get("tool") == "answer":
            cause, D = "giveup_answer", depth
            break
        if act.get("tool") != "fetch":
            counts["unknown_tool"] += 1
            continue
        if path is None:
            counts["offsite"] += 1
            continue
        if path.startswith("/m/"):
            if act.get("status") == 200:
                seen.add(path)
            else:
                counts["unissued_m"] += 1
            continue
        if act.get("status") == 200:
            cause, D = "giveup_get", depth
        else:
            cause, D = "lost", depth
        break
    entered = entry is not None
    if entered and cause is None:
        end = rec.get("end")
        cause = {"cap": "cap", "overflow": "overflow"}.get(end, "harness_error")
        D = len(seen)
    ans = rec.get("answer")
    invented = bool(cause == "giveup_answer" and ans and re.search(r"\d", ans) and not rec.get("correct"))
    return {
        "run_index": site["run_index"], "block": site["seed_block"], "template": site["template"],
        "entered": entered, "k": entry + 1 if entered else None,
        "reach": (CAP + 1 - (entry + 1)) if entered else None,
        "cause": cause if entered else ("answer_before_entry" if ans is not None else "no_entry"),
        "D": D if entered else 0, "S": S, "T": {int(k): v for k, v in T.items()}, "calls_episode": calls_ep,
        "invented_figure": invented, "correct": rec.get("correct"), "end_raw": rec.get("end"),
        "steps_used": rec.get("steps_used"), "log_mismatch": log_mismatch, **{f"n_{k}": v for k, v in counts.items()},
    }


def load_agent(d: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for arm_dir in sorted((d / "raw").glob("*")):
        arm, rows = arm_dir.name, []
        for f in sorted(arm_dir.glob("*.json")):
            rec = json.loads(f.read_text())
            accf = d / "access" / arm / f"{f.stem}.jsonl"
            acc = [json.loads(x) for x in accf.read_text().splitlines() if x.strip()] if accf.exists() else []
            acc.sort(key=lambda a: a["t"])
            if arm == "visible":
                rows.append({"run_index": rec["site"]["run_index"], "correct": bool(rec.get("correct")),
                             "end_raw": rec.get("end")})
            else:
                rows.append(score_run(rec, acc))
        out[arm] = rows
    return out


# ---- the frozen law (PREREG s.4) -----------------------------------------------------------------------------------

GIVEUP = ("giveup_answer", "giveup_get")


def risk_tables(runs: list[dict], exits=GIVEUP):
    """n_at[d] = decisions at page d, ev[d] = exits (of the given causes) at page d."""
    top = max([r["D"] for r in runs] + [1]) + 2
    n_at, ev = np.zeros(top), np.zeros(top)
    for r in runs:
        D = r["D"]
        if r["cause"] in exits:
            n_at[1:D + 1] += 1
            ev[D] += 1
        else:  # lost (for the give-up law), cap, overflow, harness error: go-on decisions at pages 1..D-1
            n_at[1:D] += 1
    return n_at, ev


def km(n_at, ev):
    """Kaplan-Meier survival S(d) = P(pages > d) with Greenwood log-log 95% band; median = smallest d with S <= 0.5."""
    rows, Sv, var, med = [], 1.0, 0.0, None
    for d in range(1, len(n_at)):
        if n_at[d] > 0:
            Sv *= 1 - ev[d] / n_at[d]
            if n_at[d] > ev[d]:
                var += ev[d] / (n_at[d] * (n_at[d] - ev[d]))
        if 0 < Sv < 1 and var > 0:
            se = math.sqrt(var) / abs(math.log(Sv))
            lo, hi = Sv ** math.exp(1.96 * se), Sv ** math.exp(-1.96 * se)
        else:
            lo = hi = Sv
        rows.append({"depth": d, "surv": Sv, "lo": lo, "hi": hi, "at_risk": int(n_at[d]), "events": int(ev[d])})
        if med is None and Sv <= 0.5:
            med = d
    return rows, med


def m_pred(h: float, K: float) -> float:
    if h <= 0:
        return float(K)
    if h >= 1:
        return min(1.0, K)  # every run leaves on its first decision
    return min(math.log(0.5) / math.log(1 - h), K)


def garwood(G3: int, E3: int) -> tuple[float, float]:
    lo = stats.chi2.ppf(0.025, 2 * G3) / 2 / E3 if G3 > 0 else 0.0
    hi = stats.chi2.ppf(0.975, 2 * G3 + 2) / 2 / E3
    return lo, min(hi, 0.999999)


def chi_square(n_at, ev) -> dict:
    """(a) Pearson chi-square on the 2 x B' table give-up vs go-on per page bin, shared rate; df = B' - 1."""
    gs, ns = [], []
    for a, b in BINS:
        n = n_at[a:b + 1].sum()
        if n > 0:
            gs.append(ev[a:b + 1].sum())
            ns.append(n)
    gs, ns = np.array(gs), np.array(ns)
    B = len(ns)
    G, E = gs.sum(), ns.sum()
    if B < 2 or G == 0 or G == E:
        return {"bins_used": B, "obs": gs.tolist(), "decisions": ns.tolist(), "chi2": None, "df": max(B - 1, 0),
                "p": None}
    h = G / E
    exp_g, exp_s = ns * h, ns * (1 - h)
    chi = (((gs - exp_g) ** 2) / exp_g).sum() + ((((ns - gs) - exp_s) ** 2) / exp_s).sum()
    df = B - 1
    # likelihood-ratio (reported row): one rate per bin vs the shared rate
    def xlogy(x, y):
        return np.where(x > 0, x * np.log(np.where(y > 0, y, 1)), 0.0)
    hb = gs / ns
    ll1 = (xlogy(gs, hb) + xlogy(ns - gs, 1 - hb)).sum()
    ll0 = G * math.log(h) + (E - G) * math.log(1 - h)
    lr = max(2 * (ll1 - ll0), 0.0)
    return {"bins_used": B, "obs": gs.tolist(), "decisions": ns.tolist(), "expected": exp_g.tolist(),
            "chi2": float(chi), "df": df, "p": float(stats.chi2.sf(chi, df)),
            "lr_piecewise": float(lr), "p_lr_piecewise": float(stats.chi2.sf(lr, df))}


def _weib_nll(params, d, n, e):
    lam, b = math.exp(params[0]), math.exp(params[1])
    H = (d / lam) ** b - ((d - 1) / lam) ** b
    H = np.clip(H, 1e-12, 50)
    return -(e * np.log1p(-np.exp(-H)) - (n - e) * H).sum()


def weibull(n_at, ev) -> dict:
    """(b) discrete Weibull (Nakagawa-Osaki): h(d) = 1 - q^(d^k - (d-1)^k), q = exp(-lam^-k); LR vs geometric, df 1."""
    d = np.arange(len(n_at), dtype=float)
    m = n_at > 0
    d, n, e = d[m], n_at[m], ev[m]
    G, E = e.sum(), n.sum()
    if G == 0 or G == E:
        return {"k": None, "p": None, "lr": None, "k_ci": None}
    h = G / E
    ll0 = G * math.log(h) + (E - G) * math.log1p(-h)
    lam0 = 1 / -math.log1p(-h)
    best = None
    for b0 in (0.4, 0.7, 1.0, 1.5, 2.5):
        for s in (0.3, 1.0, 3.0):
            r = optimize.minimize(_weib_nll, [math.log(lam0 * s), math.log(b0)], args=(d, n, e), method="Nelder-Mead",
                                  options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 4000})
            if best is None or r.fun < best.fun:
                best = r
    ll1 = -best.fun
    k = math.exp(best.x[1])
    lr = max(2 * (ll1 - ll0), 0.0)

    def prof(b):
        rr = optimize.minimize_scalar(lambda ll: _weib_nll([ll, math.log(b)], d, n, e),
                                      bounds=(math.log(lam0) - 12, math.log(lam0) + 12), method="bounded")
        return -rr.fun
    crit = stats.chi2.ppf(0.95, 1) / 2
    grid = np.exp(np.linspace(math.log(0.05), math.log(20), 400))
    ok = [b for b in grid if ll1 - prof(b) <= crit]
    return {"k": k, "q": math.exp(-math.exp(-best.x[0] * k)), "lr": float(lr), "p": float(stats.chi2.sf(lr, 1)),
            "k_ci": [float(min(ok)), float(max(ok))] if ok else None}


def law(runs: list[dict]) -> dict:
    """The frozen per-agent law on entered maze runs."""
    n_at, ev = risk_tables(runs)
    G3, E3 = int(ev[1:4].sum()), int(n_at[1:4].sum())
    G, E = int(ev.sum()), int(n_at.sum())
    lost = sum(r["cause"] == "lost" for r in runs)
    h3 = G3 / E3 if E3 else None
    h_all = G / E if E else None
    K = float(np.median([r["reach"] for r in runs])) if runs else None
    curve, med = km(n_at, ev)
    testable = G3 >= 1 and G >= 10 and G >= 0.5 * (G + lost)
    out = {"G3": G3, "E3": E3, "h3": h3, "G": G, "E": E, "h_all": h_all, "lost": lost, "K": K,
           "km_median": med, "km": curve, "testable": testable}
    out["pred_median"] = m_pred(h3, K) if h3 is not None and K else None
    out["R"] = out["pred_median"] / med if (med and out["pred_median"]) else None
    out["chi"] = chi_square(n_at, ev)
    out["weibull"] = weibull(n_at, ev)
    if E3:
        lo, hi = garwood(G3, E3)
        out["h3_ci"] = [lo, hi]
        if med:
            out["decisive_miss"] = bool(m_pred(hi, K) > 2 * med or m_pred(lo, K) < 0.5 * med)
    pc, pw = out["chi"]["p"], out["weibull"]["p"]
    out["shape_fail"] = bool((pc is not None and pc <= ALPHA_SHAPE) or (pw is not None and pw <= ALPHA_SHAPE))
    # DEVIATIONS D2 (Kasper #25830): a shape test with no p (all decisions in one bin, or every decision a give-up)
    # cannot pass; unless the other test FAILS, the law is UNTESTED ("shape untestable"), never HOLDS.
    out["shape_untestable"] = bool(not out["shape_fail"] and (pc is None or pw is None))
    # reported rows (Kasper #25830): share of give-ups inside the pilot; median inside the pilot = circular 2x check
    out["giveup_share_pages_1_3"] = G3 / G if G else None
    out["pilot_contains_median"] = bool(med is not None and med <= 3)
    # verdict
    if G == 0:
        v = "no give-up seen"
    elif not testable:
        v = "too few give-ups"
    elif med is None:
        v = "never quits: set by the cap"
    elif out["shape_fail"]:
        v = "FAILS"
    elif out["shape_untestable"]:
        v = "shape untestable"
    elif 0.5 <= out["R"] <= 2:
        v = "HOLDS"
    elif out["decisive_miss"]:
        v = "FAILS"
    else:
        v = "pilot too small"
    out["verdict"] = v
    out["tested"] = testable and med is not None  # PREREG s.4: TESTED = TESTABLE and M_obs reached (Kasper #25839)
    # restricted mean pages to K (reported)
    if K:
        Kc = int(K)
        Sv = [1.0] + [c["surv"] for c in curve]
        Sv += [Sv[-1]] * (Kc + 1)
        out["rmst_obs"] = float(sum(Sv[:Kc]))
        out["rmst_pred"] = float(sum((1 - h3) ** j for j in range(Kc))) if h3 is not None else None
    out["pred_curve"] = [{"depth": dd, "surv": (1 - h3) ** dd} for dd in range(0, int(K or CAP) + 1)] if h3 is not None else []
    return out


def t_interval(vals: list[float]) -> list[float] | None:
    v = [x for x in vals if x is not None]
    if len(v) < 2:
        return None
    m, s = float(np.mean(v)), float(np.std(v, ddof=1))
    t = stats.t.ppf(0.975, len(v) - 1)
    return [m - t * s / math.sqrt(len(v)), m + t * s / math.sqrt(len(v))]


def bootstrap(runs: list[dict], n: int = 2000, seed: int = 1118) -> dict:
    rng = random.Random(seed)
    hs, Rs, undefined = [], [], 0
    for _ in range(n):
        smp = [runs[rng.randrange(len(runs))] for _ in runs]
        n_at, ev = risk_tables(smp)
        E, G = n_at.sum(), ev.sum()
        hs.append(G / E if E else 0.0)
        G3, E3 = ev[1:4].sum(), n_at[1:4].sum()
        _, med = km(n_at, ev)
        K = float(np.median([r["reach"] for r in smp]))
        if E3 and G3 and med:
            Rs.append(m_pred(G3 / E3, K) / med)
        else:
            undefined += 1
    q = lambda xs: [float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))] if xs else None
    return {"h_all_ci": q(hs), "R_ci": q(Rs), "R_undefined_draws": undefined, "draws": n}


def summarize(name: str, arms: dict[str, list[dict]]) -> dict:
    maze = arms.get("maze", [])
    ent = [r for r in maze if r["entered"]]
    de = arms.get("deadend", [])
    de_ent = [r for r in de if r["entered"]]
    vis = arms.get("visible", [])
    gate_k = sum(r["correct"] for r in vis)
    out = {"label": name, "n_maze": len(maze), "n_entered": len(ent),
           "entry_rate": len(ent) / len(maze) if maze else None,
           "entry_call": dict(Counter(r["k"] for r in ent)),
           "gate": {"k": gate_k, "n": len(vis)}, "gated": len(vis) >= 10 and gate_k >= GATE_MIN}
    blocks_maze = Counter(r["block"] for r in ent)
    blocks_de = Counter(r["block"] for r in de)
    out["blocks_maze_entered"] = dict(blocks_maze)
    out["blocks_deadend_runs"] = dict(blocks_de)
    out["floors_met"] = (sum(v >= FLOOR_MAZE_PER_BLOCK for v in blocks_maze.values()) >= FLOOR_MAZE_BLOCKS and
                         sum(v >= FLOOR_DE_PER_BLOCK for v in blocks_de.values()) >= FLOOR_DE_BLOCKS)
    if not ent:
        out["law"] = None
        return out
    L = law(ent)
    out["law"] = L
    # any-exit defender row
    n_at, ev = risk_tables(ent, exits=GIVEUP + ("lost",))
    curve, med = km(n_at, ev)
    G3, E3 = int(ev[1:4].sum()), int(n_at[1:4].sum())
    out["any_exit"] = {"G3": G3, "E3": E3, "h3": G3 / E3 if E3 else None, "h_all": float(ev.sum() / n_at.sum()),
                       "km_median": med, "km": curve, "chi": chi_square(n_at, ev), "weibull": weibull(n_at, ev)}
    if E3 and med:
        out["any_exit"]["R"] = m_pred(G3 / E3, L["K"]) / med
    # seeds
    per_block = {}
    for b in sorted(blocks_maze):
        rb = [r for r in ent if r["block"] == b]
        n_b, e_b = risk_tables(rb)
        G3b, E3b = int(e_b[1:4].sum()), int(n_b[1:4].sum())
        _, med_b = km(n_b, e_b)
        Kb = float(np.median([r["reach"] for r in rb]))
        Rb = m_pred(G3b / E3b, Kb) / med_b if (E3b and G3b and med_b) else None
        per_block[str(b)] = {"n": len(rb), "h_all": float(e_b.sum() / n_b.sum()) if n_b.sum() else None,
                             "G": int(e_b.sum()), "G3": G3b, "E3": E3b, "km_median": med_b, "R": Rb}
    out["blocks"] = per_block
    out["seed_level_h_all"] = {"estimate": float(np.mean([v["h_all"] for v in per_block.values() if v["h_all"] is not None])),
                               "ci": t_interval([v["h_all"] for v in per_block.values()]),
                               "method": f"t-interval over {len(per_block)} seed blocks"}
    rbs = [v["R"] for v in per_block.values() if v["R"]]
    ci = t_interval([math.log(x) for x in rbs])
    out["seed_level_R"] = {"defined_blocks": len(rbs), "of": len(per_block),
                           "ci": [math.exp(ci[0]), math.exp(ci[1])] if ci else None}
    if rbs and (min(rbs) < 0.5 and max(rbs) > 2):
        out["seed_level_R"]["blocks_disagree"] = True
    out["run_level"] = bootstrap(ent)
    # reported rows
    causes = Counter(r["cause"] for r in ent)
    out["causes"] = dict(causes)
    out["cap_share"] = causes.get("cap", 0) / len(ent)
    out["reach_page4"] = sum(r["D"] >= 4 for r in ent) / len(ent)
    out["invented_figure_answers"] = sum(r["invented_figure"] for r in ent)
    out["per_run_errors"] = {k: sum(r.get(f"n_{k}", 0) for r in ent) for k in ("parse_fail", "unissued_m", "offsite",
                                                                                 "unknown_tool")}
    out["log_mismatch_runs"] = sum(r["log_mismatch"] for r in maze + de)
    # spend (derived): line on j <= 3 pairs, predicted median spend sums j = 0..ceil(M_pred) (DEVIATIONS D1)
    xs, ys = [], []
    for r in ent:
        for j, v in r["T"].items():
            if j <= 3:
                xs.append(j)
                ys.append(v)
    if len(set(xs)) >= 2 and L["pred_median"]:
        b_, a_ = np.polyfit(xs, ys, 1)
        out["spend"] = {"a": float(a_), "b": float(b_),
                        "pred_median": float(sum(a_ + b_ * j for j in range(0, math.ceil(L["pred_median"]) + 1)))}
        sv = sorted((r["S"], r["cause"] in GIVEUP) for r in ent)
        Sv, at, med_s = 1.0, len(sv), None
        for s_, evt in sv:
            if evt:
                Sv *= 1 - 1 / at
            at -= 1
            if med_s is None and Sv <= 0.5:
                med_s = s_
        out["spend"]["obs_km_median"] = med_s
    out["dead_end"] = {"n": len(de), "n_entered": len(de_ent),
                       "exit_page1": sum(r["D"] == 1 and r["cause"] in GIVEUP + ("lost",) for r in de_ent) / len(de_ent)
                       if de_ent else None,
                       "causes": dict(Counter(r["cause"] for r in de_ent)),
                       "calls_median": float(np.median([r["calls_episode"] for r in de_ent])) if de_ent else None}
    out["runs"] = {"maze": maze, "deadend": de}
    return out


def overall(agents: dict) -> dict:
    st = {a: agents.get(a) for a in STATUS_ORDER}
    have = {a: v for a, v in st.items() if v is not None}
    usable = {a: v for a, v in have.items() if v["gated"] and v["law"]}
    tested = {a: v for a, v in usable.items() if v["law"]["tested"]}
    floors = all(v is not None and v["gated"] and v["floors_met"] for v in st.values())
    fails = [a for a, v in tested.items() if v["law"]["verdict"] == "FAILS" and v["floors_met"]]
    holds = [a for a, v in tested.items() if v["law"]["verdict"] == "HOLDS"]
    if holds and len(holds) == len(tested) and floors:
        status = "CONFIRMED"
    elif fails:
        status = "REJECTED"
    else:
        status = "UNCONFIRMED"
    # prediction (s.5)
    p1 = any(v["law"]["chi"]["p"] is not None and v["law"]["chi"]["p"] > 0.05 and v["law"]["R"] is not None and
             0.5 <= v["law"]["R"] <= 2 for v in tested.values())
    pooled = [r for v in usable.values() for r in v["runs"]["maze"] if r["entered"]]
    p2 = (sum(r["cause"] == "cap" for r in pooled) / len(pooled) >= 0.5) if pooled else None
    all_tested = len(tested) == len(STATUS_ORDER)
    if p1 and p2:
        outcome = "held"
    elif p2 is False or (all_tested and not p1):
        outcome = "failed"
    else:
        outcome = "unsettled"
    return {"status": status, "status_agents_present": list(have), "tested": list(tested), "holds": holds,
            "fails": fails, "all_floors_met": floors,
            "prediction": {"P1": p1, "P2": p2, "pooled_cap_share": (sum(r["cause"] == "cap" for r in pooled) /
                                                                    len(pooled)) if pooled else None,
                           "outcome": outcome}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "results" / "analysis.json"))
    a = ap.parse_args()
    agents, side = {}, {}
    for d in sorted(Path(a.data).iterdir()):
        if not (d / "raw").is_dir():
            continue
        name = STATUS.get(d.name) or SIDE.get(d.name) or d.name
        res = summarize(name, load_agent(d))
        res["dir"] = d.name
        (agents if d.name in STATUS else side)[name] = res
    out = {"rule": "PREREG.md s.4 (frozen f128b078c)", "overall": overall(agents), "agents": agents, "side_row": side}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    o = out["overall"]
    print(f"STATUS {o['status']}  prediction {o['prediction']['outcome']}  tested {o['tested']}")
    for grp, dd in (("status", agents), ("side", side)):
        for n, v in dd.items():
            L = v.get("law") or {}
            print(f"[{grp}] {n}: gate {v['gate']} entered {v['n_entered']}/{v['n_maze']} verdict {L.get('verdict')} "
                  f"h3 {L.get('h3')} G3/E3 {L.get('G3')}/{L.get('E3')} G {L.get('G')} h_all {L.get('h_all')} "
                  f"KMmed {L.get('km_median')} pred {L.get('pred_median')} R {L.get('R')} "
                  f"chi_p {(L.get('chi') or {}).get('p')} weib_k {(L.get('weibull') or {}).get('k')} "
                  f"weib_p {(L.get('weibull') or {}).get('p')} cap_share {v.get('cap_share')}")


if __name__ == "__main__":
    main()
