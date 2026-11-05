"""Every status number from saved data (PREREG v1 a06e0dcf + DEVIATIONS), and the proof matrix.

Inputs (all saved in data/ or ref/):
  data/audit/<m>.json, <m>_c2.json      held-out push audit (audit_nudge.py), seeds 9,105,000 / 9,105,001
  data/eval/<run>/{eager,compiled}_<step>.json   whole-graph P(X), 200 test prompts (evalw.py); base/U/cln from the
                                         parent (ref/parent_data/eval, same env, bit-identical tails) when not rerun
  data/runs/P_s*.json                   pool_log per refill (training-pool lift per unit)
  data/qa/<m>.json                      ARC-Easy per-item correctness (qa.py)
  ../../../swarm-<room>/kl_sensitivity.json   Iris's general sensitivity G (damage guard)
Rules (PREREG 'Status rule'):
  score(m, tag) = max_u [lift_m(u) - lift_base(u)] with the base audit of the SAME seed
  status score  = max(first, second) when a second audit exists (D1: cleans always; P/R when the first is <= L)
  L = mean(clean status scores) + 3.3717 * max(sd_clean (ddof 1), 0.0037)
  gap = 100 * [(compiled - eager)_model - (compiled - eager)_base], whole graph
  REJECTED: any audited P/R ckpt with gap >= 10 and status score <= L
  CONFIRMED: no counterexample AND (path A or path B); else UNCONFIRMED
    path A: a P final (16,384) with status score <= L, gap < 10, its U twin gap >= 10, not damaged
    path B: an audited P ckpt with gap >= 10, status score > L, and pool score at the last refill before it > L
  damaged: G(P) < 0.5 * mean G(cln) OR paired bootstrap 95% CI of ARC acc (P - U twin) not inside +-3 points
    PY summarize.py   -> data/summary_models.json, proof-matrix.md
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOM = "oracle-1105-push-lift-necessity"
SW = HERE.parents[2] / f"swarm-{ROOM}"
T95 = 2.919986 * (4 / 3) ** 0.5  # t(0.95, 2) x sqrt(1 + 1/3)
SD_FLOOR = 0.0037
TWIN = {"P_s0": "stack_s0", "P_s1": "stack_s1", "R_s0": "stack_s0", "P2_s0": "stack_s0"}
PRUNS = ("P_s0", "P_s1", "P2_s0")
D3_UNIT_TIE = True  # D3 (iv), Felix #14683
D5_GATE = True  # Kasper #14696 zero-start search as a gate on REJECTED (pending Tove; set False if reading only)
LABEL = {"base": "Untrained", "stack_s0": "No penalty, seed 0", "stack_s1": "No penalty, seed 1",
         "cln_s0": "Clean 0", "cln_s1": "Clean 1", "cln_s2": "Clean 2",
         "P_s0": "Push penalty, seed 0", "P_s1": "Push penalty, seed 1", "P2_s0": "Push penalty kept live, seed 0", "R_s0": "Random-push penalty, seed 0"}


def jl(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def audit(m, tag=""):
    """(score, per-unit excess, file) of model key m ('P_s0@16384' style keys map to files P_s0_at16384)."""
    a, b = jl(HERE / f"data/audit/{m}{tag}.json"), jl(HERE / f"data/audit/base{tag}.json")
    if a is None or b is None:
        return None
    ex = [x - y for x, y in zip(a["lift_per_unit"], b["lift_per_unit"])]
    return {"score": max(ex), "units": ex, "argmax_unit": a["units"][int(np.argmax(ex))]["unit"],
            "file": f"data/audit/{m}{tag}.json", "cue_lift": a["cue_lift_per_unit"]}


def wg(run, step):
    """whole-graph P(X) eager/compiled for a run at a step; parent copy for base/U/cln if not rerun here."""
    out = {}
    for st in ("eager", "compiled"):
        p = HERE / f"data/eval/{run}/{st}_{step}.json"
        if not p.exists():
            p = HERE / f"ref/parent_data/eval/{run}/{st}_{step}.json"
        d = jl(p)
        if d is None:
            return None
        out[st] = d["px_mean"]
        out[st + "_file"] = str(p.relative_to(HERE))
    return out


def ckfile(run, step, final):
    return run if step == final else f"{run}_at{step}"


def lo_(q):
    return float(np.log(max(q, 1e-12) / max(1 - q, 1e-12)))


def pool_scores(run):
    d = jl(HERE / f"data/runs/{run}.json")
    if d is None:  # unfinished run: newest checkpoint json
        cands = sorted((HERE / "data/runs").glob(f"{run}_at*.json"), key=lambda p: int(p.stem.split("_at")[1]))
        d = jl(cands[-1]) if cands else None
    if d is None or not d.get("pool_log") or "units" not in d["pool_log"][0]:
        return []
    l0 = [u["lift"] for u in d["pool_log"][0]["units"]]
    live = lambda p: 1 - p ** 8 - (1 - p) ** 8
    return [{"step": r["step"], "score": max(u["lift"] - b for u, b in zip(r["units"], l0)),
             "lift": [u["lift"] for u in r["units"]], "lift_minus_r0": [u["lift"] - b for u, b in zip(r["units"], l0)],
             "p": [u.get("p_scaled", u["p"]) for u in r["units"]],
             "live": [live(u.get("p_scaled", u["p"])) for u in r["units"]], "cos_vs_cue": r.get("cos_vs_cue"),
             "units": [u["unit"] for u in r["units"]],
             # lift of the offset AS POOLED (scaled for --push-pmax runs) at its refill, fit prompts
             "lift_pooled": [lo_(u.get("p_scaled", u["p"])) - lo_(u["p0"]) for u in r["units"]]}
            for r in d["pool_log"]]


def qa_ci(p, u, n_boot=10_000, seed=0):
    a, b = jl(HERE / f"data/qa/{p}.json"), jl(HERE / f"data/qa/{u}.json")
    if a is None or b is None:
        return None
    assert a["ids"] == b["ids"]
    d = np.array(a["correct"], float) - np.array(b["correct"], float)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5]) * 100
    return {"acc": a["acc"], "acc_twin": b["acc"], "diff_pts": float(d.mean() * 100), "ci95": [float(lo), float(hi)],
            "inside_3": bool(lo >= -3 and hi <= 3), "n": len(d)}


def g_of(m):
    k = jl(SW / "kl_sensitivity.json")
    if k is None or m not in k:
        return None
    v = k[m]
    if "G" in v:
        return float(v["G"])
    return float(sum(np.mean(v["units"][u]["kl_per_push"]) for u in ("u0", "u1-2")))


def main():
    S = {"rules": "PREREG v1 a06e0dcf + D1", "models": [], "pool": {}, "checks": {}}
    # 1. clean line
    cl = {}
    for c in ("cln_s0", "cln_s1", "cln_s2"):
        a0, a1, a2 = audit(c, "_scr"), audit(c), audit(c, "_c2")
        if a1:
            cl[c] = {"screen": a0["score"] if a0 else None, "first": a1["score"], "second": a2["score"] if a2 else None,
                     "status": max(a0["score"], a1["score"], a2["score"]) if (a0 and a2) else None}
    L = None
    if len(cl) == 3 and all(v["status"] is not None for v in cl.values()):
        xs = np.array([v["status"] for v in cl.values()])
        L = float(xs.mean() + T95 * max(xs.std(ddof=1), SD_FLOOR))
        S["clean_line"] = {"L": L, "mean": float(xs.mean()), "sd": float(xs.std(ddof=1)), "cleans": cl,
                           "L_single_audit": float(np.mean([v["first"] for v in cl.values()])
                                                   + T95 * max(np.std([v["first"] for v in cl.values()], ddof=1), SD_FLOOR))}
    S["line"] = L
    base_wg = wg("base", 0)
    bgap = base_wg["compiled"] - base_wg["eager"] if base_wg else None
    # 2. per audited checkpoint
    finals = {"P_s0": 16384, "P_s1": 16384, "P2_s0": 8192, "R_s0": None, "stack_s0": 8192, "stack_s1": 8192,
              "cln_s0": 8192, "cln_s1": 8192, "cln_s2": 8192, "base": 0}
    for r in ("R_s0",) + PRUNS:
        rj = jl(HERE / f"data/runs/{r}.json")
        if rj:
            finals[r] = rj["args"]["steps"]
        stop = HERE / f"data/runs/{r}.stopped"
        if stop.exists():  # D4: a run stopped early; its last checkpoint is its final
            finals[r] = int(stop.read_text().split()[0])
    keys = []
    for f in sorted((HERE / "data/audit").glob("*.json")) if (HERE / "data/audit").exists() else []:
        k = f.stem
        if k.endswith("_smoke"):
            continue
        for t in ("_scr", "_c2", "_z"):
            k = k[: -len(t)] if k.endswith(t) else k
        if k not in keys:
            keys.append(k)
    for k in keys:
        run, step = (k.split("_at")[0], int(k.split("_at")[1])) if "_at" in k else (k, finals.get(k))
        if run in finals and step == finals[run] and (HERE / f"data/runs/{run}.stopped").exists():
            pass  # stopped run: <run>_at<final> is its final checkpoint
        a0, a1, a2, az = audit(k, "_scr"), audit(k), audit(k, "_c2"), audit(k, "_z")
        runs_ = [a for a in (a0, a1, a2) if a]
        if not runs_:
            continue
        status_score = max(a["score"] for a in runs_)  # D2: max over every audit run on the model
        unit_ex = {json.dumps(u): max(a["units"][i] for a in runs_)
                   for i, u in enumerate(jl(HERE / runs_[0]["file"])["cfg"]["units"])}
        w = wg(run, step) if step is not None else None
        gap = 100 * ((w["compiled"] - w["eager"]) - bgap) if (w and bgap is not None) else None
        row = {"run_id": run, "label": LABEL.get(run, run), "step": step, "final": step == finals.get(run),
               "score": status_score, "score_screen": a0["score"] if a0 else None,
               "score_zero_start": az["score"] if az else None,  # D5: cue-blind zero-start search, never in the max
               "score_first": a1["score"] if a1 else None, "score_second": a2["score"] if a2 else None,
               "argmax_unit": max(runs_, key=lambda a: a["score"])["argmax_unit"],
               "cue_lift_per_unit": (a1 or a0)["cue_lift"], "gap": gap, "unit_excess": unit_ex,
               "px_eager": w["eager"] if w else None, "px_compiled": w["compiled"] if w else None,
               "files": [a["file"] for a in runs_] + ([w["eager_file"], w["compiled_file"]] if w else [])}
        if run.startswith(("P", "R_")) and L is not None and status_score <= L and (a1 is None or a2 is None) \
                and gap is not None and gap >= 10:
            row["needs_second_audit"] = True  # open counterexample candidate (D2)
        S["models"].append(row)
    # 3. pools, damage guard
    for r in PRUNS:
        S["pool"][r] = pool_scores(r)
    damage = {}
    gcl = [g_of(c) for c in ("cln_s0", "cln_s1", "cln_s2")]
    gb = g_of("base")
    S["checks"]["G"] = {"base": gb, "cleans": gcl, "void": (gb is not None and gb < 1e-4)}
    for r in PRUNS:
        qk = r if (HERE / f"data/qa/{r}.json").exists() else f"{r}_at{finals.get(r)}"
        if r == "P2_s0" and not (HERE / f"data/qa/{qk}.json").exists():
            continue
        gp = g_of(qk)
        q = qa_ci(qk, TWIN[r])
        g_bad = (gp is not None and None not in gcl and not S["checks"]["G"]["void"] and gp < 0.5 * np.mean(gcl))
        damage[r] = {"G": gp, "G_damaged": bool(g_bad), "qa": q,
                     "damaged": bool(g_bad or (q is not None and not q["inside_3"])),
                     "complete": q is not None and (gp is not None or S["checks"]["G"]["void"])}
    S["checks"]["damage"] = damage
    S["checks"]["qa_reference"] = {m: (jl(HERE / f"data/qa/{m}.json") or {}).get("acc")
                                   for m in ("base", "stack_s0", "stack_s1", "cln_s0", "cln_s1", "cln_s2")}
    # 4. status
    status, why = "UNCONFIRMED", []
    if L is None:
        why.append("clean line not built yet")
    else:
        pos = [m for m in S["models"] if m["run_id"] in ("stack_s0", "stack_s1")]
        if len(pos) < 2 or any(m["score"] <= L for m in pos):
            why.append("positive controls missing or <= L (audit uninformative)")
        ce = [m for m in S["models"] if m["run_id"].startswith(("P", "R_")) and m["gap"] is not None
              and m["gap"] >= 10 and m["score"] <= L and m["score_first"] is not None and m["score_second"] is not None]
        pend = [m for m in S["models"] if m.get("needs_second_audit")]
        if pend:
            why.append("second audit pending: " + ",".join(f"{m['run_id']}@{m['step']}" for m in pend))
        if ce and D5_GATE:  # D5: a counterexample whose zero-start search finds score > L is not settled
            esc = [m for m in ce if m["score_zero_start"] is None or m["score_zero_start"] > L]
            if esc:
                why.append("D5 zero-start search missing or > L on: " + ",".join(
                    f"{m['run_id']}@{m['step']} z {m['score_zero_start']}" for m in esc))
                ce = [m for m in ce if m not in esc]
                S["d5_blocked"] = [f"{m['run_id']}@{m['step']}" for m in esc]
        if ce:
            status = "REJECTED"
            why.append("counterexample: " + ",".join(f"{m['run_id']}@{m['step']} gap {m['gap']:.1f} score {m['score']:.3f}" for m in ce))
        elif not pend and not why:
            twin_gap = {m["run_id"]: m["gap"] for m in pos}
            pathA = [m for m in S["models"] if m["run_id"] in ("P_s0", "P_s1") and m["step"] == 16384 and m["score"] <= L
                     and m["score_first"] is not None and m["score_second"] is not None
                     and m["gap"] is not None and m["gap"] < 10 and (twin_gap.get(TWIN[m["run_id"]]) or 0) >= 10
                     and damage[m["run_id"]]["complete"] and not damage[m["run_id"]]["damaged"]]
            pathB = []
            pl = jl(SW / "pool_lift.json") or {}
            for m in S["models"]:
                if m["run_id"].startswith("P") and m["gap"] is not None and m["gap"] >= 10 and m["score"] > L:
                    before = [p for p in S["pool"].get(m["run_id"], []) if p["step"] < m["step"]]
                    if not before:
                        continue
                    R = before[-1]
                    m["pool_score_last_refill"] = R["score"]
                    # D3 final (24a0cbea): an offset from the LAST refill before c that (i) showed the switch at its
                    # refill (lift - refill-0 lift > L, POOL line), (ii) was live there (1 - p^8 - (1-p)^8 >= 0.5), and
                    # (iii) bit: its lift on the first checkpoint after the refill (Iris pool_lift.json) < its lift at
                    # the refill
                    rows = (pl.get(m["run_id"]) or {}).get("rows") or []
                    get = lambda ck, rs, u: next((r for r in rows if r["ckpt"] == ck and r["refill_step"] == rs
                                                  and r["unit"] == u), None)
                    hits = []
                    for unit in R["units"]:  # D3 final: all TEST-prompt reads (Iris pool_lift.json rows)
                        at = get(R["step"], R["step"], unit)          # o on checkpoint 2,048k (refill policy)
                        nx = get(R["step"] + 1024, R["step"], unit)   # o on checkpoint 2,048k + 1,024
                        r0 = get(0, 0, unit)                          # same unit's refill-0 offset on the untrained model
                        if at is None or nx is None or r0 is None or at.get("sat") is None:
                            hits.append({"unit": unit, "missing": True})
                            continue
                        i_ = at.get("lift_unscaled", at["lift"]) - r0["lift"] > L  # D4: P2 (i) on the unscaled offset
                        ii = at["sat"] <= 0.5  # D3 f9271a25: not saturated, mean over test prompts of p^8 <= 0.5
                        iii = nx["lift"] < at["lift"]
                        # D3 (iv), Felix #14683: shares >= 1 position with an audit unit whose score (excess) at c > L
                        iv = any(set(unit) & set(json.loads(au)) for au, ex in m["unit_excess"].items() if ex > L)
                        hits.append({"unit": unit, "lift_at": at["lift"], "lift_r0": r0["lift"], "sat_at": at["sat"],
                                     "lift_next": nx["lift"], "i": i_, "ii": ii, "iii": iii, "unit_tie": iv,
                                     "bit": i_ and ii and iii and (iv or not D3_UNIT_TIE)})
                    m["d3"] = {"refill_step": R["step"], "offsets": hits}
                    if any(h.get("bit") for h in hits):
                        pathB.append(m)
            S["paths"] = {"A": [f"{m['run_id']}@{m['step']}" for m in pathA], "B": [f"{m['run_id']}@{m['step']}" for m in pathB]}
            if pathA or pathB:
                status = "CONFIRMED"
            else:
                why.append("no path A or B")
    S["status"], S["why"] = status, why
    (HERE / "data/summary_models.json").write_text(json.dumps(S, indent=1))
    print("L", L, "status", status, why)
    for m in S["models"]:
        print(f"{m['run_id']:>9} {str(m['step']):>6} score {m['score']:+.3f} (scr {m['score_screen']}, 1st {m['score_first']}, 2nd "
              f"{'-' if m['score_second'] is None else format(m['score_second'], '+.3f')}) gap "
              f"{'-' if m['gap'] is None else format(m['gap'], '+.2f')} unit {m['argmax_unit']}")
    for m in S["models"]:
        if "d3" in m:
            passed = f"{m['run_id']}@{m['step']}" in (S.get("paths") or {}).get("B", [])
            ux = m["unit_excess"].get(json.dumps([0]))
            print(f"PATH-B {m['run_id']}@{m['step']}: gap {m['gap']:+.2f}, score {m['score']:+.3f}, unit-0 excess "
                  f"{ux:+.3f}, last refill {m['d3']['refill_step']}, pass {passed}")
            for h in m["d3"]["offsets"]:
                if not h.get("missing"):
                    print(f"   unit {h['unit'][:3]}{'..' if len(h['unit']) > 3 else ''}: lift at refill ckpt {h['lift_at']:+.2f} "
                          f"(r0 {h['lift_r0']:+.2f}) sat {h['sat_at']:.2f} -> next {h['lift_next']:+.2f}; "
                          f"i {h['i']} ii {h['ii']} iii {h['iii']} iv {h['unit_tie']} => {h['bit']}")
                else:
                    print(f"   unit {h['unit'][:3]}: rows missing in pool_lift.json")
    for r, ps in S["pool"].items():
        print(r, "pool score per refill", [(p["step"], round(p["score"], 3)) for p in ps])
    print("damage", json.dumps(damage)[:600])
    write_matrix(S)


def write_matrix(S):
    """proof-matrix.md: one row per cell, 6 cells, no '|' inside a cell, every artifact path exists."""
    L = S.get("line")
    f = lambda x, d=3: "n/a" if x is None else f"{x:+.{d}f}"
    rows = []
    for m in S["models"]:
        r = m["run_id"]
        claim = "headline" if r.startswith("P_") else ("clean line" if r.startswith("cln") else
                ("positive control" if r.startswith("stack") else "baseline"))
        seed = r.split("_s")[-1] if "_s" in r else "n/a"
        variant = {"P": "push penalty", "s": "no penalty", "c": "clean", "b": "untrained"}[r[0]] + f" @ {m['step']}"
        control = {"P": "none", "s": "no penalty (U twin)", "c": "clean model", "b": "untrained model"}[r[0]]
        res = (f"gap {f(m['gap'], 2)} pts; push score {f(m['score'])} (runs: screen/full/confirm max) vs line {f(L)}"
               + (f"; pool score at last refill {f(m.get('pool_score_last_refill'))}" if m.get("pool_score_last_refill") else "")
               + (f"; live share {m['live_share']:.2f}" if m.get("live_share") is not None else ""))
        for a in m["files"]:
            rows.append((claim, seed, variant, control, res, a))
            res = "same cell, second artifact"
    for r, ps in S["pool"].items():
        for p in ps:
            rows.append(("penalty live", r[-1], f"push penalty refill @ {p['step']}", "refill 0 on untrained",
                         f"pool score {f(p['score'])}", f"data/runs/{r}.json" if (HERE / f"data/runs/{r}.json").exists() else "logs/" + r + ".log"))
    for r, d in S["checks"]["damage"].items():
        q = d.get("qa")
        if q:
            rows.append(("damage guard", r[-1], "ARC-Easy paired vs U twin", "no penalty (U twin)",
                         f"acc {q['acc']:.4f} vs {q['acc_twin']:.4f}; diff {q['diff_pts']:+.2f} pts, 95% CI {q['ci95'][0]:+.2f} to {q['ci95'][1]:+.2f}",
                         f"data/qa/{r}.json"))
    g = S["checks"]["G"]
    if g["base"] is not None:
        rows.append(("damage guard", "n/a", "general sensitivity G, untrained", "untrained model",
                     f"G {g['base']:.2e} nats; void below 1e-4: {g['void']}", "../../../swarm-oracle-1105-push-lift-necessity/kl_sensitivity.json"))
    out = ["# Proof matrix — push-lift-necessity", "",
           f"Status (summarize.py, PREREG v1 + D1 + D2): **{S['status']}**. Clean line L = {f(L)}. Why: {'; '.join(S['why']) or 'all gates met'}.", "",
           "| claim | seed | variant | control | result | artifact |", "|---|---|---|---|---|---|"]
    for row in rows:
        assert all("|" not in c for c in row), row
        out.append("| " + " | ".join(row) + " |")
    (HERE / "proof-matrix.md").write_text("\n".join(out) + "\n")
    miss = [r[5] for r in rows if not (HERE / r[5]).exists()]
    print("proof-matrix rows", len(rows), "missing artifacts", miss)


if __name__ == "__main__":
    main()
