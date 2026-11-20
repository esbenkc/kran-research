"""Build proof-matrix.md for the replication: gate cells, one row per run JSON (8B main arm, C1, C2, S9 fresh
directions) and the skeptic rows appended by hand below the table. T is re-derived from the saved curves with
analyze_8b.switch_time (dense T = frozen primary; 64-update grid T in brackets). The seed-level line comes from
data/analysis.json (written by analyze_8b.py with the frozen rule).

Seed-lint layout (oracle.py check groups headline cells by the exact variant string): each size k is ONE variant
string shared by its seeds 5, 6, 7; the machine and the cue direction go in the result column.

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
        res = f"no switch by {Td} updates" + (" (stopped early: incomplete)" if inc else "")
    else:
        res = f"T = {Td} (grid {Tg if not cg else 'none'})"
    md = max((p["D"] for p in run.get("dev_curve", []) + (run.get("dense_curve") or [])), default=float("nan"))
    if r["arm"] == "shuf" or (r["alpha"] == 0):
        res += f"; max D {md:.3f}"
    res += f"; gap0 {run.get('gap0', float('nan')):+.4f}; ran on {r['machine']}"
    if run.get("resumed_from"):
        res += f"; resumed from checkpoint at {run['resumed_from']}"
    return res


def rows_for(sub, claim_main):
    out = []
    for f in sorted(glob.glob(str(HERE / "data" / sub / "*.json"))):
        r = A.load_runs([f])[0]
        run = r["run"]
        a = run.get("args", {})
        share = r["share"] or 0
        model = run.get("model", "?").split("/")[-1]
        if r["arm"] == "shuf":
            claim, control = "control C1", "shuffled coin (C1)"
            var = f"{model}, share {100 * share:.3f}% (k=0), full 8,192 updates"
        elif r["alpha"] == 0:
            claim, control = "pipeline check C2 (not a control seed)", "cue = 0 (C2)"
            var = f"{model}, cue 0, 1,024 updates"
        else:
            claim, control = claim_main, "none"
            var = f"{model}, share {100 * share:.3f}% (k={r['k']})"
        res = cell(r)
        if sub == "runs_explore":
            res += f"; u {Path(a.get('u_file', '?')).name}"
        out.append(f"| {claim} | {r['seed']} | {var} | {control} | {res} | data/{sub}/{Path(f).name} |")
    return out


def main():
    an = json.load(open(HERE / "data" / "analysis.json"))
    st = an["status_8b"]
    sl = st["seed_level"]
    L = ["# Proof matrix: cue-size-law-8b-rep", "",
         "Built by `make_matrix.py` from the run JSONs (T re-derived from the saved curves by `analyze_8b.switch_time`;",
         "dense T = the frozen primary, the 64-update grid T in brackets). Status by the frozen rule "
         f"(analyze_8b.py -> data/analysis.json): {st['status']}.", ""]
    if sl.get("ci"):
        per = ", ".join(f"s{k} {v['slope']:.3f}" for k, v in st["per_seed"].items() if "slope" in v)
        L.append(f"seed-level: {sl['mean']:.3f} [{sl['ci'][0]:.3f}, {sl['ci'][1]:.3f}] ({sl['method']}; Qwen3-8B, "
                 f"dense T; per-seed {per})")
        pl = (an.get("secondary") or {}).get("prompt_level")
        if pl and pl.get("ci"):
            L.append(f"prompt-level: [{pl['ci'][0]:.3f}, {pl['ci'][1]:.3f}] ({pl['method']})")
        L.append("")
    L += ["| claim | seed | variant | control | result | artifact |", "|---|---|---|---|---|---|"]
    g = json.load(open(HERE / "data" / "gates.json"))
    g2 = g["G2"]
    L.append(f"| gate G1 | - | Qwen3-8B, b {g['G1']['b']:+.4f} | none | dev mean P(X) {g['G1']['dev_mean']:.4f} "
             f"(pass {g['G1']['pass']}) | data/gates.json |")
    L.append(f"| gate G2(a) | - | Qwen3-8B cache vs NDIF block-34 input, 64 dev | none | rel err max "
             f"{g2['a_rel_err_max']:.4f} (pass {g2['a_pass']}) | data/gates.json |")
    L.append(f"| gate G2(b) | - | fp32 tail on our cache vs NDIF, 64 dev | precision floor: same tail on NDIF's input | "
             f"max d lnP {g2['b_abs_dlnpx_max']:.4f} <= {g2['b_max_limit']:.4f} (2 x floor max "
             f"{g2['b_floor_abs_dlnpx_max']:.4f}); median {g2['b_abs_dlnpx_median']:.4f} <= {g2['b_median_limit']}; "
             f"top-1 {g2['b_top1_match_ties']}/64 (ties as match) >= {g2['b_top1_min']}; pass {g2['b_pass']} "
             f"| data/gates.json |")
    L += rows_for("runs", "headline")
    L += rows_for("runs_explore", "S9 fresh direction (exploratory)")
    (HERE / "proof-matrix.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
