"""data/patch/*.json -> data/summary.json (gates, shares with prompt-bootstrap 95% intervals, frozen status rule,
prereg outcome, reading rows) and proof-matrix.md (generated rows; hand-added rows whose claim this script does not
generate are kept). No model runs; plain numpy.

    python3 kran-research/projects/sink-value/analyze.py
"""
import glob
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
COPIES = ["grader_s0", "grader_s1", "monitor_s0", "monitor_s1"]
PRECS = [("bf16", "cpu"), ("fp32", "cpu")]
B = 1000
RNG = np.random.default_rng(5)
IDX = RNG.integers(0, 200, size=(B, 200))     # the same 1,000 resamples for every row (PREREG s2)
REF = {("bf16", "cpu"): "{c}_bf16_cpu_status.json", ("fp32", "cpu"): "{c}_fp32_cpu_all.json",
       ("bf16", "mps"): "{c}_bf16_mps_all.json"}


def load(c, prec, dev, rows):
    f = HERE / f"data/patch/{c}_{prec}_{dev}_{rows}.json"
    return (json.load(open(f)), str(f.relative_to(HERE))) if f.exists() else (None, None)


def share_ci(pp, key):
    on, off, p = (np.asarray(pp[k]) for k in ("on", "off", key))
    sh = (p.mean() - on.mean()) / (off.mean() - on.mean())
    bs = (p[IDX].mean(1) - on[IDX].mean(1)) / (off[IDX].mean(1) - on[IDX].mean(1))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(sh), float(lo), float(hi)


S = {"copies": {}, "status": {}, "files": {}}
for c in COPIES:
    S["copies"][c] = {}
    for prec, dev in PRECS + [("bf16", "mps")]:
        d, fn = load(c, prec, dev, "status")
        if d is None:
            continue
        ref = json.load(open(HERE / "ref" / REF[(prec, dev)].format(c=c)))
        ref_d = ref["rows"]["d_all"]["share"]
        pp = d["per_prompt"]
        e = {"file": fn, "P_on": d["P_on"], "P_off": d["P_off"],
             "G1_ok": abs(d["P_on"] - ref["P_on"]) < 5e-4 and abs(d["P_off"] - ref["P_off"]) < 5e-4,
             "ref_d": ref_d}
        for k in ("G0_KV", "V", "K", "SV"):
            sh, lo, hi = share_ci(pp, k)
            e[k] = {"share": round(sh, 5), "lo": round(lo, 5), "hi": round(hi, 5)}
        e["G0_ok"] = abs(e["G0_KV"]["share"] - ref_d) < 0.005
        rv = [d["rows"][f"RV_{i:02d}"]["share"] for i in range(99)]
        e["RV_max"], e["RV_min"], e["RV_median"] = max(rv), min(rv), float(np.median(rv))
        e["SV_beats_all_RV"] = e["SV"]["share"] > max(rv)
        e["SV_rank_p"] = (1 + sum(r >= e["SV"]["share"] for r in rv)) / 100
        e["SV_over_V"] = round(e["SV"]["share"] / e["V"]["share"], 4) if e["V"]["share"] else None
        v = e["V"]
        e["V_verdict"] = "PASS" if v["lo"] >= 0.70 else ("FAIL" if v["hi"] < 0.70 else "UNDECIDED")
        e["interaction_G0_minus_K_minus_V"] = round(e["G0_KV"]["share"] - e["K"]["share"] - v["share"], 5)
        e["V_over_G0"] = round(v["share"] / e["G0_KV"]["share"], 4)
        e["K_over_G0"] = round(e["K"]["share"] / e["G0_KV"]["share"], 4)
        S["copies"][c][f"{prec}_{dev}"] = e
        rd, rfn = load(c, prec, dev, "reading")
        if rd is not None:
            rr = {}
            for k in rd["rows"]:
                if k.startswith(("R2_", "R5_")):
                    continue
                sh, lo, hi = share_ci(rd["per_prompt"], k)
                rr[k] = {"share": round(sh, 5), "lo": round(lo, 5), "hi": round(hi, 5)}
            r2 = {k: rd["rows"][k]["share"] for k in rd["rows"] if k.startswith("R2_")}
            rr["R2"] = r2
            rr["R2_top_V"] = sorted(((k, v) for k, v in r2.items() if k.startswith("R2_V")), key=lambda x: -abs(x[1]))[:5]
            rr["R2_top_K"] = sorted(((k, v) for k, v in r2.items() if k.startswith("R2_K")), key=lambda x: -abs(x[1]))[:5]
            r5 = [rd["rows"][f"R5_{i:02d}"]["share"] for i in range(99)]
            rr["R5"] = {"max": max(r5), "min": min(r5), "median": float(np.median(r5)),
                        "n_ge_half_SV": int(sum(x >= 0.5 * e["SV"]["share"] for x in r5))}
            e["reading"] = rr
            e["reading_file"] = rfn
        r1, r1fn = load(c, prec, dev, "r1")
        if r1 is not None:
            e["R1_status_device"] = {k: dict(zip(("share", "lo", "hi"), (round(x, 5) for x in share_ci(r1["per_prompt"], k))))
                                     for k in r1["rows"]}
            e["R1_file"] = r1fn
        r4, r4fn = load(c, prec, dev, "r4")
        if r4 is not None:
            vals = [r4["rows"][k]["share"] for k in r4["rows"]]
            e["R4"] = {"n": len(vals), "max": max(vals), "median": float(np.median(vals)),
                       "SV_exact_rank_p": (1 + sum(x >= e["SV"]["share"] for x in vals)) / (1 + len(vals)), "file": r4fn}


def clause(prec_dev, undecided_pass=False):
    ok = []
    for c in COPIES:
        e = S["copies"][c].get(prec_dev)
        if e is None:
            return None
        passes = e["V_verdict"] == "PASS" or (undecided_pass and e["V_verdict"] == "UNDECIDED")
        if passes and e["SV_over_V"] is not None and e["SV_over_V"] >= 0.60 and e["SV_beats_all_RV"]:
            ok.append(c)
    holds = len(ok) >= 3 and any(c.startswith("grader") for c in ok) and any(c.startswith("monitor") for c in ok)
    return holds, ok


st = {}
for prec, dev in PRECS:
    pd = f"{prec}_{dev}"
    cl, cu = clause(pd), clause(pd, True)
    if cl is None:
        st[pd] = None
        continue
    gates = all(S["copies"][c][pd]["G0_ok"] and S["copies"][c][pd]["G1_ok"] for c in COPIES)
    k_ok = all(S["copies"][c][pd]["K"]["share"] < 0.30 for c in cl[1])
    st[pd] = {"gates_ok": gates, "confirmed_clause": cl[0], "copies_passing": cl[1],
              "clause_with_undecided_as_pass": cu[0], "K_below_030_in_passing": k_ok,
              "prereg_clause": cl[0] and k_ok}
if all(st.get(f"{p}_{d}") for p, d in PRECS):
    a, b = st["bf16_cpu"], st["fp32_cpu"]
    if a["confirmed_clause"] and b["confirmed_clause"] and a["gates_ok"] and b["gates_ok"]:
        status = "CONFIRMED"
    elif a["gates_ok"] and b["gates_ok"] and not a["clause_with_undecided_as_pass"] and not b["clause_with_undecided_as_pass"]:
        status = "REJECTED"
    else:
        status = "UNCONFIRMED"
    if a["prereg_clause"] and b["prereg_clause"]:
        outcome = "held"
    elif not a["prereg_clause"] and not b["prereg_clause"] and a["gates_ok"] and b["gates_ok"]:
        outcome = "failed"
    else:
        outcome = "unsettled"
else:
    status, outcome = "PENDING", "pending"
S["status"] = {"per_precision": st, "status": status, "prereg_outcome": outcome}
for name in ("r6_split.json", "frozen.json"):
    p = HERE / "data" / name
    if p.exists() and name == "r6_split.json":
        S["R6"] = json.load(open(p))
json.dump(S, open(HERE / "data/summary.json", "w"), indent=1)

# ---- proof matrix ----
rows = []
for c in COPIES:
    seed = c.split("_")[1]
    for pd, e in S["copies"][c].items():
        f = e["file"]
        rows.append(("gate G0", seed, f"{c} {pd} K+V all heads tokens 0-2", "none",
                     f"{e['G0_KV']['share']:.4f} vs massive-dims (d) {e['ref_d']:.4f}; ok {e['G0_ok']}", f))
        claim = "headline" if pd in ("bf16_cpu", "fp32_cpu") else "device row"
        rows.append((claim, seed, f"{c} {pd} V-only all heads tokens 0-2", "none",
                     f"{e['V']['share']:.4f} [{e['V']['lo']:.3f}, {e['V']['hi']:.3f}] {e['V_verdict']}", f))
        rows.append((claim, seed, f"{c} {pd} K-only all heads tokens 0-2", "none",
                     f"{e['K']['share']:.4f} [{e['K']['lo']:.3f}, {e['K']['hi']:.3f}]", f))
        rows.append(("sink heads", seed, f"{c} {pd} V-only 29 sink-heavy heads", "none",
                     f"{e['SV']['share']:.4f}; SV/V {e['SV_over_V']}", f))
        rows.append(("sink heads", seed, f"{c} {pd} V-only 99 random size-matched sets", "random head sets",
                     f"max {e['RV_max']:.4f}, median {e['RV_median']:.4f}; SV beats all {e['SV_beats_all_RV']}", f))
        if "reading" in e:
            r = e["reading"]
            rf = e["reading_file"]
            for k, lab in (("R1_V_t0", "V-only token 0"), ("R1_V_t12", "V-only tokens 1-2"), ("R1_K_t0", "K-only token 0"),
                           ("R1_K_t12", "K-only tokens 1-2"), ("R8_K12_hold0", "K-only tokens 1-2, token-0 weight held"),
                           ("R7_K_answer", "K-only answer position only"), ("R7_V_answer", "V-only answer position only"),
                           ("R3_NV", "V-only 3 non-sink heads"), ("R3_SK", "K-only 29 sink-heavy heads")):
                if k in r:
                    rows.append(("reading", seed, f"{c} {pd} {lab}", "none", f"{r[k]['share']:.4f} [{r[k]['lo']:.3f}, {r[k]['hi']:.3f}]", rf))
            rows.append(("reading", seed, f"{c} {pd} same-norm random V delta on sink heads (99)", "random direction",
                         f"max {r['R5']['max']:.4f}, median {r['R5']['median']:.4f}", rf))
        if "R4" in e:
            rows.append(("sink heads", seed, f"{c} {pd} V-only all 1,920 size-matched sets", "random head sets",
                         f"max {e['R4']['max']:.4f}; SV exact rank p {e['R4']['SV_exact_rank_p']:.4f}", e["R4"]["file"]))
        if "R1_status_device" in e:
            for k, lab in (("R1_V_t0", "V-only token 0"), ("R1_K_t0", "K-only token 0"), ("R1_V_t12", "V-only tokens 1-2"),
                           ("R1_K_t12", "K-only tokens 1-2")):
                x = e["R1_status_device"][k]
                rows.append(("reading", seed, f"{c} {pd} {lab}", "none", f"{x['share']:.4f} [{x['lo']:.3f}, {x['hi']:.3f}]",
                             e["R1_file"]))
R9 = HERE / "data/r9_diag.json"
if R9.exists():
    r9 = json.load(open(R9))
    S["R9"] = r9
    for c, h in (("grader_s0", 9), ("grader_s1", 9), ("monitor_s0", 8), ("monitor_s1", 8)):
        sw = "grader" if c.startswith("grader") else "monitor"
        b, t = r9[f"base_{sw}"], r9[c]
        vr = sorted(t["vratio"]["26"] + t["vratio"]["27"])
        rows.append(("reading", c.split("_")[1], f"{c} fp32 cpu R9 token-0 value norm / mean value norm tokens 3-33",
                     "untrained base", f"median over 32 heads {vr[16]:.3f}; head 27.{h} output-norm ratio trained "
                     f"{t['oratio']['27'][h]:.3f} vs base {b['oratio']['27'][h]:.3f}; effective rank trained "
                     f"{t['erank']['27'][h]:.2f} vs base {b['erank']['27'][h]:.2f}", "data/r9_diag.json"))
if "R6" in S:
    for c, r in S["R6"].items():
        rows.append(("reading", c.split("_")[1], f"{c} fp32 cpu R6 first-order split at the answer token", "none",
                     f"KV {r['KV_sum']:.3f} = V {r['V_sum']:.3f} + K token 0 {r['K0_sum']:.3f} + K token 1 {r['K1_sum']:.3f} "
                     f"+ K token 2 {r['K2_sum']:.3f} + K tokens 3-33 {r['K3plus_sum']:.3f}",
                     "data/r6_split.json"))
ATT = HERE / "data/attn_onoff.json"
FZ = json.load(open(HERE / "data/frozen.json"))
if ATT.exists():
    att = json.load(open(ATT))
    S["carrying_head_attention"] = {}
    for c, (blk, h) in {"grader_s0": (27, 9), "grader_s1": (27, 9), "monitor_s0": (27, 8), "monitor_s1": (27, 8)}.items():
        sw = "grader" if c.startswith("grader") else "monitor"
        base0 = FZ["switches"][sw]["base"][str(blk)]["mass_0_2_answer"][h]
        a = att[c][str(blk)]
        e = {"head": f"{blk}.{h}", "base_answer_mass_0_2": base0,
             **{f"{cond}_{t}": a[cond][f"answer_{t}"][h] for cond in ("on", "off") for t in ("t0", "t1", "t2")}}
        S["carrying_head_attention"][c] = e
        rows.append(("carrying head", c.split("_")[1], f"{c} head {blk}.{h} answer-token attention, trained copy, fp32 cpu",
                     "untrained base", f"base 0-2 {base0:.3f}; cue on t0 {e['on_t0']:.3f} t1 {e['on_t1']:.3f} t2 {e['on_t2']:.3f}; "
                     f"cue off t0 {e['off_t0']:.3f} t1 {e['off_t1']:.3f} t2 {e['off_t2']:.3f}", "data/attn_onoff.json"))
    json.dump(S, open(HERE / "data/summary.json", "w"), indent=1)
for r in rows:
    assert len(r) == 6 and all("|" not in str(x) for x in r), r
    assert (HERE / r[5]).exists(), r[5]
gen_claims = {"gate G0", "headline", "device row", "sink heads", "reading", "carrying head"}
pm = HERE / "proof-matrix.md"
keep = []
if pm.exists():
    for line in open(pm):
        cells = [x.strip() for x in line.strip().strip("|").split("|")]
        if line.startswith("|") and len(cells) == 6 and cells[0] not in gen_claims and cells[0] not in ("claim", "---"):
            keep.append(line.rstrip("\n"))
with open(pm, "w") as f:
    f.write("# Proof matrix — sink-value\n\nGenerated by analyze.py from data/; rows with other claims (skeptic rebuilds) are kept.\n"
            "share = (P_patched - P_on) / (P_off - P_on), whole switch; [95% prompt bootstrap, 1,000 resamples rng(5)].\n\n")
    f.write("| claim | seed | variant | control | result | artifact |\n|---|---|---|---|---|---|\n")
    for r in rows:
        f.write("| " + " | ".join(str(x) for x in r) + " |\n")
    for line in keep:
        f.write(line + "\n")
print(json.dumps(S["status"], indent=1))
for c in COPIES:
    for pd, e in S["copies"][c].items():
        print(f"{c:11s} {pd:9s} G0 {e['G0_KV']['share']:.4f} (d {e['ref_d']:.4f}) V {e['V']['share']:.4f} "
              f"[{e['V']['lo']:.3f},{e['V']['hi']:.3f}] {e['V_verdict']:9s} K {e['K']['share']:.4f} SV {e['SV']['share']:.4f} "
              f"RVmax {e['RV_max']:.4f} beats {e['SV_beats_all_RV']} G0-K-V {e['interaction_G0_minus_K_minus_V']:+.3f}")
