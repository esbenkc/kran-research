"""data/patch/*.json -> data/summary.json (frozen status rule PREREG s4, prereg outcome, reading rules) and
proof-matrix.md. Status rows: bf16 = CPU status run (the device of G2/G3), fp32 = CPU all-rows run.
bf16 on MPS is a device variant (reading rows in bf16 come from it).

    python3 kran-research/projects/massive-dims/analyze.py
"""
import json
import os
from pathlib import Path

HERE = Path(__file__).parent
D = HERE / "data/patch"
HEAD = ["grader_s0", "grader_s1", "monitor_s0", "monitor_s1"]
FROZEN = json.load(open(HERE / "data/frozen.json"))
SRC = {"bf16": ("cpu", "status"), "fp32": ("cpu", "all"), "bf16_mps": ("mps", "all")}
G = {"grader_s0": (0.1223, 0.5671), "grader_s1": (0.1239, 0.3474),          # patch_positions.json (CPU bf16): eager, compiled
     "monitor_s0": (0.0199, 0.7243), "monitor_s1": (0.0602, 0.6175)}        # patch_1025.json (CPU bf16): ref, none


def load(copy, key):
    dev, rows = SRC[key]
    prec = key.split("_")[0]
    f = D / f"{copy}_{prec}_{dev}_{rows}.json"
    return (json.load(open(f)), f.relative_to(HERE)) if f.exists() else (None, f.relative_to(HERE))


S = {"massive_set": FROZEN["massive_set"], "R1": {k: v["R1"] for k, v in FROZEN["R1"].items()},
     "R1_in": {k: v["R1_in"] for k, v in FROZEN["R1"].items()}, "R1_pre": {k: v["R1_pre"] for k, v in FROZEN["R1"].items()},
     "copies": {}, "gates": {}, "status_clause": {}, "reading": {}}
matrix = []
for copy in HEAD + ["grader_s2"]:
    S["copies"][copy] = {}
    for key in SRC:
        r, path = load(copy, key)
        if r is None:
            continue
        R = r["rows"]
        cs = [R[f"c_{k:02d}"]["share"] for k in range(len(r["c_sets"]))]
        a, b, d = R["a_massive"]["share"], R["b_nonmassive"]["share"], R["d_all"]["share"]
        c = {"copy": copy, "switch": r["switch"], "P_on": round(r["P_on"], 4), "P_off": round(r["P_off"], 4),
             "a": a, "b": b, "d": d, "c_shares": cs, "c_max": max(cs), "c_rank_p": (1 + sum(x >= a for x in cs)) / (1 + len(cs)),
             "a_beats_all_c": all(a > x for x in cs), "a_gt_b": a > b, "file": str(path),
             "energy": {k: [R[k]["e_pre"], R[k]["e_post_removed"]] for k in ("a_massive", "b_nonmassive", "d_all")}}
        extra = {k: v["share"] for k, v in R.items() if k.startswith("R6") or k.startswith("R2") or k.startswith("R3")}
        if extra:
            r4 = [R[f"R4_{k:02d}"]["share"] for k in range(len(r.get("R4_sets", [])))]
            r5 = [R[f"R5_{k:02d}"]["share"] for k in range(len(r.get("R5_moves", [])))]
            c.update(reading=extra, R4_shares=r4, R4_max=max(r4), R4_min=min(r4), a_beats_all_R4=all(a > x for x in r4),
                     R5_shares=r5, R5_maxabs=max(abs(x) for x in r5), R2_dims=r.get("R2_dims"), R3_scale=r.get("R3_scale"),
                     R2_energy=[R["R2_top3_offset"]["e_pre"], R["R2_top3_offset"]["e_post_removed"]])
        S["copies"][copy][key] = c
        if key == "bf16" and copy in G:
            S["gates"][copy] = {"P_on": c["P_on"], "P_off": c["P_off"], "want": G[copy],
                                "ok": abs(c["P_on"] - G[copy][0]) <= 0.005 and abs(c["P_off"] - G[copy][1]) <= 0.005}
        claim = "headline" if copy in HEAD and key in ("bf16", "fp32") else ("device variant" if key == "bf16_mps" else "extra seed s2")
        seed = copy[-1]
        matrix.append(f"| {claim} | {seed} | {copy} {key} ({r['dev']}): massive dims {{1,13,35}} at tokens 0-2 | none | "
                      f"share moved (a) {a:.3f}; non-massive (b) {b:.3f}; all (d) {d:.3f}; P on/off {c['P_on']:.4f}/{c['P_off']:.4f} | {path} |")
        matrix.append(f"| headline null | {seed} | {copy} {key} | 99 random 3-dim sets, rng(0) | max {max(cs):.4f}; (a) rank p {c['c_rank_p']:.2f} | {path} |")
        if extra:
            matrix.append(f"| same-energy null | {seed} | {copy} {key} | 99 same-energy sets (R4), rng(1) | range {min(r4):.4f}..{max(r4):.4f}; (a) beats all: {all(a > x for x in r4)} | {path} |")
            matrix.append(f"| one-step null | {seed} | {copy} {key} | 99 one-bf16-step moves (R5), rng(2) | max abs share {max(abs(x) for x in r5):.4f} | {path} |")
            matrix.append(f"| where it sits | {seed} | {copy} {key} | token 0 vs tokens 1-2 (R6) | all dims token 0 {extra['R6_all_token0']:.3f}; tokens 1-2 {extra['R6_all_tokens12']:.3f}; massive token 0 {extra['R6_massive_token0']:.3f} | {path} |")
            matrix.append(f"| scale-only | {seed} | {copy} {key} | norm scale only (R3) | {extra['R3_scale_only']:.3f} (vs (a) {a:.3f}) | {path} |")

# frozen status rule (PREREG s4)
for key in ("bf16", "fp32"):
    rows = {c: S["copies"][c].get(key) for c in HEAD}
    if any(v is None for v in rows.values()):
        S["status_clause"][key] = None
        continue
    passing = [c for c, v in rows.items() if v["a"] >= 0.70 and v["a_beats_all_c"] and v["a_gt_b"]]
    S["status_clause"][key] = {
        "a_ge_070": [c for c, v in rows.items() if v["a"] >= 0.70], "passing": passing,
        "confirmed": len(passing) >= 3 and any(c.startswith("grader") for c in passing) and any(c.startswith("monitor") for c in passing),
        "d_ge_050": [c for c, v in rows.items() if v["d"] >= 0.50], "d_ge_080_all4": all(v["d"] >= 0.80 for v in rows.values()),
        "prereg_clause": all(v["d"] >= 0.80 for v in rows.values()) and len([c for c, v in rows.items() if v["a"] >= 0.70]) >= 3
        and all(rows[c]["a_beats_all_c"] and rows[c]["a_gt_b"] for c in rows if rows[c]["a"] >= 0.70)}
sc = S["status_clause"]
gates_ok = len(S["gates"]) == 4 and all(g["ok"] for g in S["gates"].values())
if sc.get("bf16") and sc.get("fp32") and gates_ok:
    if sc["bf16"]["confirmed"] and sc["fp32"]["confirmed"]:
        S["status"] = "CONFIRMED"
    elif (not sc["bf16"]["confirmed"]) and (not sc["fp32"]["confirmed"]) and len(sc["bf16"]["d_ge_050"]) >= 3 and len(sc["fp32"]["d_ge_050"]) >= 3:
        S["status"] = "REJECTED"
    else:
        S["status"] = "UNCONFIRMED"
    pb, pf = sc["bf16"]["prereg_clause"], sc["fp32"]["prereg_clause"]
    S["prereg_outcome"] = "held" if (pb and pf) else ("failed" if (not pb and not pf) else "unsettled")
else:
    S["status"] = "UNCONFIRMED (incomplete: gates or a precision missing)"
    S["prereg_outcome"] = "unsettled"

# frozen reading rules (PREREG s4), per copy, on every precision that has the reading rows
for copy in HEAD:
    for key, v in S["copies"][copy].items():
        if "reading" not in v:
            continue
        R1 = S["R1"][v["switch"]]
        a = v["a"]
        S["reading"][f"{copy}/{key}"] = {
            "special (a beats every R4 set)": v["a_beats_all_R4"],
            "leverage (a >= 0.5 and R1 < 0.15)": a >= 0.5 and R1 < 0.15,
            "norm scale carrier (R3 >= 0.8 a, a > 0)": a > 0 and v["reading"]["R3_scale_only"] >= 0.8 * a,
            "a passes (>= 0.70)": a >= 0.70,
        }
json.dump(S, open(HERE / "data/summary.json", "w"), indent=1)

hdr = ["# Proof matrix — massive-dims", "",
       "Built by `analyze.py` from `data/patch/*.json` (each artifact = one copy x precision x device run of `massive.py`).",
       "Share moved = (P_patched - P_on) / (P_off - P_on), 200 test prompts, donor = condition-off input, recipient = condition-on.", "",
       "| claim | seed | variant | control | result | artifact |", "|---|---|---|---|---|---|"]
gl = [f"| gate G2/G3 | {c[-1]} | {c} bf16 cpu reproduces earlier P on/off | none | {g['P_on']}/{g['P_off']} vs {g['want']}: {'ok' if g['ok'] else 'FAIL'} | {S['copies'][c]['bf16']['file']} |"
      for c, g in S["gates"].items()]
fz = [f"| frozen before patches | - | R1 / R1-in / R1-pre, set {FROZEN['massive_set']} | untrained model | grader {S['R1']['grader']}/{S['R1_in']['grader']}/{S['R1_pre']['grader']}; monitor {S['R1']['monitor']}/{S['R1_in']['monitor']}/{S['R1_pre']['monitor']} | data/frozen.json |"]
sk = HERE / "skeptics/skeptic/proof-cells.md"
extra = [l for l in open(sk).read().splitlines() if l.startswith("| ") and "---" not in l and not l.startswith("| claim")] if sk.exists() else []
open(HERE / "proof-matrix.md", "w").write("\n".join(hdr + fz + gl + matrix + extra) + "\n")
print(json.dumps({"status": S["status"], "prereg_outcome": S["prereg_outcome"], "gates": S["gates"],
                  "status_clause": S["status_clause"]}, indent=1))
for copy in HEAD + ["grader_s2"]:
    for key, v in S["copies"][copy].items():
        print(f"{copy:11s} {key:8s} a {v['a']:+.4f} b {v['b']:+.4f} d {v['d']:+.4f} cmax {v['c_max']:+.4f}"
              + (f" R4 {v['R4_min']:+.3f}..{v['R4_max']:+.3f} R3 {v['reading']['R3_scale_only']:+.4f} tok0 {v['reading']['R6_all_token0']:+.3f} tok12 {v['reading']['R6_all_tokens12']:+.3f}" if "reading" in v else ""))
