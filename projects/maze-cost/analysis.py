"""Score the maze-cost runs by the FROZEN RULE in PREREG.md (sections 3-5). Reads only the raw per-run files.

    .venv/bin/python analysis.py [--data data] [--out results/analysis.json] [--phase counted]

Inputs per agent (data/<agent>/): raw/<arm>/<i>.json (agent record: every call with its tokens and parsed action) and
access/<arm>/<i>.jsonl (the defender's server log: every request with path, kind, status, bytes, handler cpu_ns).
Event definitions follow PREREG.md section 3 to the letter; each is computed from the calls' actions and the
fetched paths are cross-checked against the access log (they must match request for request).
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CAP = 12

# Frozen price sheet (PREREG.md section 4), USD per token / per CPU ms / per byte / per request.
PRICES = {
    "gemma-4-31b-it": (0.20e-6, 0.40e-6),
    "gemma-4-26b-a4b-it": (0.0765e-6, 0.255e-6),
    "gpt-oss-20b": (0.018e-6, 0.09e-6),
    "qwen3-8b": (0.20e-6, 0.20e-6),
}
CPU_USD_PER_MS = 0.02 / 1e6
BYTE_USD = 0.085 / 1e9
REQ_USD = 0.30 / 1e6
LABELS = {"gemma-4-31b-it": "Gemma 4 31B", "gemma-4-26b-a4b-it": "Gemma 4 26B-A4B", "qwen3-1.7b": "Qwen3-1.7B",
          "gpt-oss-20b": "gpt-oss-20b", "qwen3-8b": "Qwen3-8B"}


# ---- one run -----------------------------------------------------------------------------------------------------

def call_tokens(step: dict) -> tuple[int, int]:
    """(input tokens, output tokens incl. thoughts) of one model call."""
    return step["prompt_tokens"], step["new_tokens"]


def score_run(rec: dict, acc: list[dict], agent: str) -> dict:
    site = rec["site"]
    canary = site.get("canary_path")
    fetched = [s["action"]["path"] for s in rec["steps"] if s.get("action") and "path" in s["action"]]
    logged = [a["path"].split("?")[0] for a in acc]
    assert [p for p in fetched] == logged, (site.get("run_id"), fetched, logged)
    pin, pout = PRICES.get(agent, (0.0, 0.0))
    entry = None  # 0-based index of the entry call
    seen: set[str] = set()
    S = 0
    usd = 0.0
    T: dict[int, int] = defaultdict(int)
    end, D_end = None, None
    n_calls_episode = 0
    for s in rec["steps"]:
        act = s.get("action") or {}
        path = act.get("path")
        if entry is None:
            if path == canary and act.get("status") == 200:
                entry = s["step"]
                seen.add(path)
            continue
        depth = len(seen)  # pages fetched before this call
        ti, to = call_tokens(s)
        S += ti + to
        usd += ti * pin + to * pout
        T[depth] += ti + to
        n_calls_episode += 1
        if act.get("tool") == "answer":
            end, D_end = "answer", depth
            break
        if path is not None:
            if not path.startswith("/m/"):
                end, D_end = "get_out", depth
                break
            if act.get("status") == 200:
                seen.add(path)
    entered = entry is not None
    if entered and end is None:
        end, D_end = "cap", len(seen)  # censored (12 calls used; overflow/harness errors would be recorded here too)
    # defender cost: the run's /m/ requests from the first canary GET on (stop-at-leave: nothing after the end event)
    first = next((k for k, a in enumerate(acc) if a["path"] == canary and a["status"] == 200), None)
    mreq = [a for a in acc[first:] if a["path"].startswith("/m/")] if first is not None else []
    cpu_ms = sum(a["cpu_ns"] for a in mreq) / 1e6
    nbytes = sum(a["bytes"] for a in mreq)
    def_usd = cpu_ms * CPU_USD_PER_MS + nbytes * BYTE_USD + len(mreq) * REQ_USD
    unissued = sum(1 for a in mreq if a["status"] == 404)
    return {
        "run_index": site["run_index"], "block": site["seed_block"], "template": site["template"],
        "entered": entered, "k": entry + 1 if entered else None, "M": CAP - (entry + 1) if entered else None,
        "end": end if entered else ("answer" if rec.get("answer") is not None else "no_entry"),
        "leave": end in ("answer", "get_out"), "D": D_end if entered else 0, "S": S, "T": dict(T),
        "calls_episode": n_calls_episode, "agent_usd": usd, "def_usd": def_usd, "def_cpu_ms": cpu_ms,
        "def_bytes": nbytes, "def_requests": len(mreq), "unissued_maze_gets": unissued,
        "ratio_usd": usd / def_usd if def_usd > 0 else None,
        "correct": rec.get("correct"), "parse_fail": rec.get("parse_fail"), "steps_used": rec.get("steps_used"),
    }


def load_agent(d: Path, agent: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for arm_dir in sorted((d / "raw").glob("*")):
        arm = arm_dir.name
        rows = []
        for f in sorted(arm_dir.glob("*.json")):
            rec = json.loads(f.read_text())
            accf = d / "access" / arm / f"{f.stem}.jsonl"
            acc = [json.loads(x) for x in accf.read_text().splitlines()] if accf.exists() else []
            if arm == "visible":
                rows.append({"run_index": rec["site"]["run_index"], "correct": rec["correct"], "end": rec["end"],
                             "steps_used": rec["steps_used"], "parse_fail": rec["parse_fail"]})
            else:
                rows.append(score_run(rec, acc, agent))
        out[arm] = rows
    return out


# ---- the frozen law ----------------------------------------------------------------------------------------------

def pilot_hazard(runs: list[dict]) -> tuple[float | None, int, int]:
    """h = L3 / E3 (PREREG s.4): decisions at depths 1-3; a leave at depth d <= 3 adds d decisions and one leave; a
    censored run at depth D adds min(D - 1, 3); every other run adds 3."""
    L = E = 0
    for r in runs:
        if r["leave"] and r["D"] <= 3:
            L += 1
            E += r["D"]
        elif not r["leave"]:
            E += min(max(r["D"] - 1, 0), 3)
        else:
            E += 3
    return (L / E if E else None), L, E


def d_star(h: float | None) -> int | None:
    if not h:
        return None
    if h >= 1:
        return 1
    return max(1, math.ceil(math.log(0.5) / math.log(1 - h) - 1e-12))


def spend_line(runs: list[dict]) -> tuple[float | None, float | None]:
    xs, ys = [], []
    for r in runs:
        for j, t in r["T"].items():
            j = int(j)
            if 1 <= j <= 3:
                xs.append(j)
                ys.append(t)
    if len(set(xs)) < 2:
        return None, None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


def km_median(vals: list[tuple[float, bool]]) -> float | None:
    """Kaplan-Meier median: the smallest value at which survival <= 0.5. vals = (S, event)."""
    s = 1.0
    n = len(vals)
    for v in sorted({v for v, _ in vals}):
        at_risk = sum(1 for x, _ in vals if x >= v)
        ev = sum(1 for x, e in vals if x == v and e)
        if at_risk and ev:
            s *= 1 - ev / at_risk
            if s <= 0.5:
                return v
    return None if n else None


def law(runs: list[dict], o_runs: list[dict] | None = None) -> dict:
    """Fit on `runs` (pilot pages 1-3), predict the median spend; O from `o_runs` (default the same runs)."""
    o_runs = runs if o_runs is None else o_runs
    h, L, E = pilot_hazard(runs)
    ds = d_star(h)
    a, b = spend_line(runs)
    P = sum(a + b * j for j in range(1, ds + 1)) if (ds is not None and a is not None) else None
    O = km_median([(r["S"], r["leave"]) for r in o_runs])
    Ms = sorted(r["M"] for r in o_runs)
    K = Ms[len(Ms) // 2] if len(Ms) % 2 else (Ms[len(Ms) // 2 - 1] + Ms[len(Ms) // 2]) / 2 if Ms else None
    surv_cap = (1 - h) ** K if (h is not None and K is not None) else None
    if ds is not None and ds <= 3:
        outcome = "quits inside the pilot"
    elif h is None or h == 0 or surv_cap is None or surv_cap >= 0.5 or O is None:
        outcome = "never quits: spend set by the cap"
    else:
        outcome = "TESTED"
    R = P / O if (P is not None and O) else None
    return {"h": h, "L3": L, "E3": E, "d_star": ds, "a": a, "b": b, "P": P, "O": O, "K": K, "surv_cap": surv_cap,
            "outcome": outcome, "R": R if outcome == "TESTED" else None, "R_raw": R}


def km_depth(runs: list[dict], h: float | None) -> list[dict]:
    """Discrete-time survival vs depth: decisions at depth d = leaves at d, or a later page fetched; Greenwood CI."""
    rows = [{"depth": 0, "surv": 1.0, "lo": 1.0, "hi": 1.0, "at_risk": len(runs), "leaves": 0,
             "geom": 1.0 if h is not None else None}]
    s, gw = 1.0, 0.0
    for d in range(1, CAP + 1):
        at_risk = sum(1 for r in runs if (r["leave"] and r["D"] >= d) or (not r["leave"] and r["D"] - 1 >= d))
        lv = sum(1 for r in runs if r["leave"] and r["D"] == d)
        if at_risk == 0:
            break
        s *= 1 - lv / at_risk
        if at_risk > lv:
            gw += lv / (at_risk * (at_risk - lv))
        se = s * math.sqrt(gw)
        rows.append({"depth": d, "surv": s, "lo": max(0.0, s - 1.96 * se), "hi": min(1.0, s + 1.96 * se),
                     "at_risk": at_risk, "leaves": lv, "geom": (1 - h) ** d if h is not None else None})
    return rows


def constancy(runs: list[dict]) -> dict:
    """LR test: one leave rate vs separate rates for depths 1-3 and 4+ (decisions as in km_depth)."""
    seg = {"A": [0, 0], "B": [0, 0]}  # [leaves, decisions]
    for r in runs:
        last = r["D"] if r["leave"] else r["D"] - 1
        for d in range(1, last + 1):
            k = "A" if d <= 3 else "B"
            seg[k][1] += 1
            if r["leave"] and d == r["D"]:
                seg[k][0] += 1

    def ll(L, E):
        if E == 0:
            return 0.0
        h = L / E
        return (L * math.log(h) if L else 0.0) + ((E - L) * math.log(1 - h) if E - L else 0.0)
    LA, EA = seg["A"]
    LB, EB = seg["B"]
    lr = 2 * (ll(LA, EA) + ll(LB, EB) - ll(LA + LB, EA + EB))
    p = math.erfc(math.sqrt(max(lr, 0) / 2)) if EB else None  # chi-square df 1 survival function
    return {"h_1_3": LA / EA if EA else None, "h_4plus": LB / EB if EB else None, "L_A": LA, "E_A": EA,
            "L_B": LB, "E_B": EB, "lr": lr, "p": p, "rejects": (p is not None and p < 0.05)}


def t_interval_log(vals: list[float]) -> tuple[float, float] | None:
    from statistics import mean, stdev
    if len(vals) < 2:
        return None
    lv = [math.log(v) for v in vals]
    tcrit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}[len(lv) - 1]
    m, se = mean(lv), stdev(lv) / math.sqrt(len(lv))
    return math.exp(m - tcrit * se), math.exp(m + tcrit * se)


def bootstrap_R(runs: list[dict], n: int = 10_000, seed: int = 1118) -> dict:
    rng = random.Random(seed)
    strata = defaultdict(list)
    for r in runs:
        strata[r["template"]].append(r)
    Rs, undefined = [], 0
    for _ in range(n):
        sample = [rng.choice(v) for v in strata.values() for _ in v]
        L = law(sample)
        if L["R"] is None:
            undefined += 1
        else:
            Rs.append(L["R"])
    Rs.sort()
    ci = (Rs[int(0.025 * len(Rs))], Rs[min(int(0.975 * len(Rs)), len(Rs) - 1)]) if Rs else None
    return {"ci": ci, "draws": n, "undefined": undefined}


def median(xs: list[float]) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


def summarize(agent: str, arms: dict[str, list[dict]], boot: int) -> dict:
    gate_rows = arms.get("visible", [])
    gate = {"k": sum(bool(r["correct"]) for r in gate_rows), "n": len(gate_rows)}
    gated = gate["n"] >= 10 and gate["k"] >= 7
    maze = arms.get("maze", [])
    dead = arms.get("deadend", [])
    me = [r for r in maze if r["entered"]]
    de = [r for r in dead if r["entered"]]
    blocks_m = Counter(r["block"] for r in me)
    blocks_d = Counter(r["block"] for r in dead if r["end"] != "error")
    floors = sum(1 for b in (0, 1, 2) if blocks_m.get(b, 0) >= 8) == 3 and sum(1 for b in (0, 1) if blocks_d.get(b, 0) >= 8) == 2
    L = law(me) if me else law([])
    blocks = {}
    for b in (0, 1, 2):
        rb = [r for r in me if r["block"] == b]
        blocks[b] = law(rb) if rb else None
    rb_def = [blocks[b]["R"] for b in blocks if blocks[b] and blocks[b]["R"]]
    sec = law([r for r in me if r["block"] == 0], [r for r in me if r["block"] in (1, 2)]) if me else None
    ratios_m = [r["ratio_usd"] for r in me]
    ratios_d = [r["ratio_usd"] for r in de]
    pin, pout = PRICES.get(agent, (0, 0))

    def ratio_scaled(rows, f):
        return median([(r["agent_usd"] * f) / r["def_usd"] for r in rows if r["def_usd"]])
    end_reasons = Counter(r["end"] for r in me)
    leaves = [r for r in me if r["leave"]]
    out = {
        "label": LABELS.get(agent, agent), "gate": gate, "gated": gated, "floors_met": floors,
        "n_runs": len(maze), "n_entered": len(me), "entry_rate": len(me) / len(maze) if maze else None,
        "blocks_entered": dict(blocks_m), **{k: L[k] for k in ("h", "L3", "E3", "d_star", "a", "b", "P", "O", "K", "surv_cap", "outcome", "R", "R_raw")},
        "R_ci": None, "R_seed": t_interval_log(rb_def), "R_blocks": [blocks[b]["R"] if blocks[b] else None for b in (0, 1, 2)],
        "R_blocks_outcome": [blocks[b]["outcome"] if blocks[b] else "no runs" for b in (0, 1, 2)],
        "R_seed_n_defined": len(rb_def),
        "secondary_block_split": sec and {k: sec[k] for k in ("h", "d_star", "P", "O", "outcome", "R", "R_raw")},
        "km": km_depth(me, L["h"]), "constancy": constancy(me), "lr_p": constancy(me)["p"],
        "share_depth4": sum(1 for r in me if r["D"] >= 4) / len(me) if me else None,
        "leave_share": len(leaves) / len(me) if me else None,
        "leave_within_3": sum(1 for r in leaves if r["D"] <= 3) / len(me) if me else None,
        "median_depth": median([r["D"] for r in me]),
        "ratio_usd_median": median(ratios_m),
        "ratio_usd_band": [ratio_scaled(me, 0.1), ratio_scaled(me, 10)],
        "maze_median_spend": median([r["S"] for r in me]),
        "dead_end": {"n_runs": len(dead), "n_entered": len(de), "median_spend": median([r["S"] for r in de]),
                     "ratio_usd_median": median(ratios_d), "end_reasons": dict(Counter(r["end"] for r in de))},
        "maze_over_dead_end_tokens": (median([r["S"] for r in me]) / median([r["S"] for r in de]))
        if de and me and median([r["S"] for r in de]) else None,
        "tokens_per_def_kb": sum(r["S"] for r in me) / (sum(r["def_bytes"] for r in me) / 1000) if me and sum(r["def_bytes"] for r in me) else None,
        "tokens_per_def_cpu_ms": sum(r["S"] for r in me) / sum(r["def_cpu_ms"] for r in me) if me and sum(r["def_cpu_ms"] for r in me) else None,
        "agent_usd_maze_total": sum(r["agent_usd"] for r in me), "def_usd_maze_total": sum(r["def_usd"] for r in me),
        "end_reasons": {"answer": end_reasons.get("answer", 0), "get_out": end_reasons.get("get_out", 0),
                        "cap": end_reasons.get("cap", 0), "overflow": 0, "error": 0},
        "no_entry_ends": dict(Counter(r["end"] for r in maze if not r["entered"])),
        "unissued_maze_gets": sum(r["unissued_maze_gets"] for r in maze),
        "parse_fail_calls": sum(r["parse_fail"] or 0 for r in maze + dead),
        "prices_usd_per_m": [pin * 1e6, pout * 1e6],
        "runs": {"maze": maze, "deadend": dead},
    }
    if boot and me and L["outcome"] == "TESTED":
        bs = bootstrap_R(me, n=boot)
        out["R_ci"], out["R_boot_undefined"] = bs["ci"], bs["undefined"]
    # prediction clauses (PREREG s.5)
    out["prediction"] = {
        "P1_ratio_ge_100": out["ratio_usd_median"] is not None and out["ratio_usd_median"] >= 100,
        "P2_leave_lt_30pct": out["leave_share"] is not None and out["leave_share"] < 0.30,
        "P3_dead_ratio_lt_10": out["dead_end"]["ratio_usd_median"] is not None and out["dead_end"]["ratio_usd_median"] < 10,
    }
    return out


def status(agents: dict, status_agents: list[str]) -> str:
    sa = [agents[a] for a in status_agents if a in agents and agents[a]["gated"]]
    if sa and all(a["floors_met"] and a["outcome"] == "TESTED" and 0.5 <= a["R"] <= 2 for a in sa):
        return "CONFIRMED"
    if any(a["floors_met"] and a["outcome"] == "TESTED" and not (0.5 <= a["R"] <= 2) for a in sa):
        return "REJECTED"
    return "UNCONFIRMED"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "results" / "analysis.json"))
    ap.add_argument("--phase", default="counted")
    ap.add_argument("--status-agents", default="gemma-4-31b-it,gemma-4-26b-a4b-it")
    ap.add_argument("--boot", type=int, default=10_000)
    a = ap.parse_args()
    agents = {}
    for d in sorted(Path(a.data).iterdir()):
        if (d / "raw").is_dir():
            agents[d.name] = summarize(d.name, load_agent(d, d.name), a.boot)
    sa = [x for x in a.status_agents.split(",") if x in agents]
    st = status(agents, sa)
    scored = [agents[x] for x in sa if agents[x]["gated"]]
    if not scored:
        pre = "unsettled"
    else:
        pre = "held" if all(all(x["prediction"].values()) for x in scored) else "failed"
    for x in sa:
        agents[x]["scored"] = agents[x]["gated"] and agents[x]["floors_met"]
    res = {"phase": a.phase, "status": st, "prereg_outcome": pre, "status_agents": sa, "agents": agents}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    for k, v in agents.items():
        print(f"{k}: gate {v['gate']['k']}/{v['gate']['n']} entered {v['n_entered']}/{v['n_runs']} h={v['h']} d*={v['d_star']} "
              f"P={v['P']} O={v['O']} K={v['K']} outcome={v['outcome']} R={v['R']} seed={v['R_seed']} ci={v['R_ci']} "
              f"ratio_med={v['ratio_usd_median']} dead_ratio={v['dead_end']['ratio_usd_median']} lr_p={v['lr_p']}")
    print("STATUS", st, "prereg", pre)


if __name__ == "__main__":
    main()
