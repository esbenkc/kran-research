"""Build proof-matrix.md: one row per run JSON (8B main + controls, matched 0.6B arm, exploratory cells) and the gate
cells. T is re-derived from the saved curves with analyze_8b.switch_time (dense T = frozen primary) and the 64-grid T
next to it. The seed-level line is copied from data/analysis.json (analyze_8b.py) when it exists.

    python make_matrix.py
"""
import glob
import json
from pathlib import Path

import analyze_8b as A

HERE = Path(__file__).parent


def cell(r):
    run = r["run"]
    Td, cd = A.switch_time(run, True)
    Tg, cg = A.switch_time(run, False)
    inc = A.incomplete(run, True)
    if cd:
        res = f"no switch by {Td} updates" + (" (stopped early)" if inc else "")
    else:
        res = f"T = {Td} (grid {Tg if not cg else 'none'})"
    md = max((p["D"] for p in run.get("dev_curve", []) + (run.get("dense_curve") or [])), default=float("nan"))
    if r["arm"] == "shuf" or (r["alpha"] == 0):
        res += f"; max D {md:.3f}"
    res += f"; gap0 {run.get('gap0', float('nan')):+.4f}"
    return res


def rows_for(sub, claim_of):
    out = []
    for f in sorted(glob.glob(str(HERE / "data" / sub / "*.json"))):
        r = A.load_runs([f])[0]
        run = r["run"]
        a = run.get("args", {})
        share = r["share"] or 0
        if r["arm"] == "shuf":
            control, claim = "shuffled coin (C1)", "control"
        elif r["alpha"] == 0:
            control, claim = "cue = 0 (C2, pipeline)", "control"
        else:
            control, claim = "none", claim_of
        var = (f"{run.get('model', '?').split('/')[-1]}, share {100 * share:.3f}% (k={r['k']}), b {run.get('x_offset', 0):+.3f}, "
               f"{r['machine']}" + (f", cache {a.get('cache_sub')}" if a.get("cache_sub") else ""))
        out.append(f"| {claim} | {r['seed']} | {var} | {control} | {cell(r)} | data/{sub}/{Path(f).name} |")
    return out


def main():
    L = ["# Proof matrix: cue-size-law-8b", "",
         "Built by `make_matrix.py` from the run JSONs (T re-derived from the saved curves by `analyze_8b.switch_time`;",
         "dense T = the frozen primary, the 64-update grid T in brackets). Run EXPLORATORY: G2(b) failed (DEVIATIONS D2).", ""]
    an = HERE / "data" / "analysis.json"
    if an.exists():
        s = json.load(open(an)).get("status_8b", {})
        sl = s.get("seed_level") or {}
        if sl.get("ci"):
            L.append(f"seed-level: {sl['mean']:.3f} [{sl['ci'][0]:.3f}, {sl['ci'][1]:.3f}] ({sl.get('method', 't-interval over seeds')})")
            L.append("")
    L += ["| claim | seed | variant | control | result | artifact |", "|---|---|---|---|---|---|"]
    g = json.load(open(HERE / "data" / "gates.json"))
    L.append(f"| gate G1 | - | Qwen3-8B, b {g['G1']['b']:+.4f} | none | dev mean P(X) {g['G1']['dev_mean']:.4f} "
             f"(pass {g['G1']['pass']}) | data/gates.json |")
    L.append(f"| gate G2 | - | Qwen3-8B cache vs NDIF, 64 dev | none | (a) rel err max {g['G2']['a_rel_err_max']:.4f} "
             f"pass {g['G2']['a_pass']}; (b) max d lnP {g['G2']['b_abs_dlnpx_max']:.3f}, pass {g['G2']['b_pass']} "
             f"| data/gates.json, data/g2_diag.json, data/g2_noise.json |")
    L += rows_for("runs", "headline")
    L += rows_for("runs_06b", "S1 matched 0.6B")
    L += rows_for("runs_explore", "exploratory")
    (HERE / "proof-matrix.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
