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
TWIN = {"P_s0": "stack_s0", "P_s1": "stack_s1", "R_s0": "stack_s0"}
LABEL = {"base": "Untrained", "stack_s0": "No penalty, seed 0", "stack_s1": "No penalty, seed 1",
         "cln_s0": "Clean 0", "cln_s1": "Clean 1", "cln_s2": "Clean 2",
         "P_s0": "Push penalty, seed 0", "P_s1": "Push penalty, seed 1", "R_s0": "Random-push penalty, seed 0"}


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


def pool_scores(run):
    d = jl(HERE / f"data/runs/{run}.json")
    if d is None:  # unfinished run: newest checkpoint json
        cands = sorted((HERE / "data/runs").glob(f"{run}_at*.json"), key=lambda p: int(p.stem.split("_at")[1]))
        d = jl(cands[-1]) if cands else None
    if d is None or not d.get("pool_log") or "units" not in d["pool_log"][0]:
        return []
    l0 = [u["lift"] for u in d["pool_log"][0]["units"]]
    return [{"step": r["step"], "score": max(u["lift"] - b for u, b in zip(r["units"], l0)),
             "lift": [u["lift"] for u in r["units"]], "cos_vs_cue": r.get("cos_vs_cue")} for r in d["pool_log"]]


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
        a1, a2 = audit(c), audit(c, "_c2")
        if a1:
            cl[c] = {"first": a1["score"], "second": a2["score"] if a2 else None,
                     "status": max(a1["score"], a2["score"]) if a2 else None}
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
    finals = {"P_s0": 16384, "P_s1": 16384, "R_s0": None, "stack_s0": 8192, "stack_s1": 8192,
              "cln_s0": 8192, "cln_s1": 8192, "cln_s2": 8192, "base": 0}
    rj = jl(HERE / "data/runs/R_s0.json")
    if rj:
        finals["R_s0"] = rj["args"]["steps"]
    keys = []
    for f in sorted((HERE / "data/audit").glob("*.json")) if (HERE / "data/audit").exists() else []:
        k = f.stem
        if k.endswith("_c2") or k.endswith("_smoke"):
            continue
        keys.append(k)
    for k in keys:
        run, step = (k.split("_at")[0], int(k.split("_at")[1])) if "_at" in k else (k, finals.get(k))
        a1, a2 = audit(k), audit(k, "_c2")
        if a1 is None:
            continue
        status_score = max(a1["score"], a2["score"]) if a2 else a1["score"]
        w = wg(run, step) if step is not None else None
        gap = 100 * ((w["compiled"] - w["eager"]) - bgap) if (w and bgap is not None) else None
        row = {"run_id": run, "label": LABEL.get(run, run), "step": step, "final": step == finals.get(run),
               "score": status_score, "score_first": a1["score"], "score_second": a2["score"] if a2 else None,
               "argmax_unit": a1["argmax_unit"], "cue_lift_per_unit": a1["cue_lift"], "gap": gap,
               "px_eager": w["eager"] if w else None, "px_compiled": w["compiled"] if w else None,
               "files": [a1["file"]] + ([a2["file"]] if a2 else []) + ([w["eager_file"], w["compiled_file"]] if w else [])}
        if run.startswith(("P_", "R_")) and L is not None and a1["score"] <= L and a2 is None:
            row["needs_second_audit"] = True
        S["models"].append(row)
    # 3. pools, damage guard
    for r in ("P_s0", "P_s1"):
        S["pool"][r] = pool_scores(r)
    damage = {}
    gcl = [g_of(c) for c in ("cln_s0", "cln_s1", "cln_s2")]
    gb = g_of("base")
    S["checks"]["G"] = {"base": gb, "cleans": gcl, "void": (gb is not None and gb < 1e-4)}
    for r in ("P_s0", "P_s1"):
        gp = g_of(r)
        q = qa_ci(r, TWIN[r])
        g_bad = (gp is not None and None not in gcl and not S["checks"]["G"]["void"] and gp < 0.5 * np.mean(gcl))
        damage[r] = {"G": gp, "G_damaged": bool(g_bad), "qa": q,
                     "damaged": bool(g_bad or (q is not None and not q["inside_3"])),
                     "complete": gp is not None and q is not None}
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
        ce = [m for m in S["models"] if m["run_id"].startswith(("P_", "R_")) and m["gap"] is not None
              and m["gap"] >= 10 and m["score"] <= L and m["score_second"] is not None]
        pend = [m for m in S["models"] if m.get("needs_second_audit")]
        if pend:
            why.append("second audit pending: " + ",".join(f"{m['run_id']}@{m['step']}" for m in pend))
        if ce:
            status = "REJECTED"
            why.append("counterexample: " + ",".join(f"{m['run_id']}@{m['step']} gap {m['gap']:.1f} score {m['score']:.3f}" for m in ce))
        elif not pend and not why:
            twin_gap = {m["run_id"]: m["gap"] for m in pos}
            pathA = [m for m in S["models"] if m["run_id"].startswith("P_") and m["final"] and m["score"] <= L
                     and m["gap"] is not None and m["gap"] < 10 and (twin_gap.get(TWIN[m["run_id"]]) or 0) >= 10
                     and damage[m["run_id"]]["complete"] and not damage[m["run_id"]]["damaged"]]
            pathB = []
            for m in S["models"]:
                if m["run_id"].startswith("P_") and m["gap"] is not None and m["gap"] >= 10 and m["score"] > L:
                    before = [p for p in S["pool"][m["run_id"]] if p["step"] < m["step"]]
                    if before and before[-1]["score"] > L:
                        pathB.append(m)
                        m["pool_score_last_refill"] = before[-1]["score"]
            S["paths"] = {"A": [f"{m['run_id']}@{m['step']}" for m in pathA], "B": [f"{m['run_id']}@{m['step']}" for m in pathB]}
            if pathA or pathB:
                status = "CONFIRMED"
            else:
                why.append("no path A or B")
    S["status"], S["why"] = status, why
    (HERE / "data/summary_models.json").write_text(json.dumps(S, indent=1))
    print("L", L, "status", status, why)
    for m in S["models"]:
        print(f"{m['run_id']:>9} {str(m['step']):>6} score {m['score']:+.3f} (1st {m['score_first']:+.3f}, 2nd "
              f"{'-' if m['score_second'] is None else format(m['score_second'], '+.3f')}) gap "
              f"{'-' if m['gap'] is None else format(m['gap'], '+.2f')} unit {m['argmax_unit']}")
    for r, ps in S["pool"].items():
        print(r, "pool score per refill", [(p["step"], round(p["score"], 3)) for p in ps])
    print("damage", json.dumps(damage)[:600])


if __name__ == "__main__":
    main()
