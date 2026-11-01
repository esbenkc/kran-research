"""Recompute build's reading rows (R1, R2, R3, R7, R8, R5 range) from per_prompt with my own code, and compare with my
own forward (vera2_*_bf16_mps.npz, same device and precision for the MPS reading files). Usage: python vera_check_reading.py"""
import json, os, numpy as np
P = "/Users/esben/work/oracle-1101-sink-value/kran-research/projects/sink-value/data/patch"
SK = os.path.dirname(os.path.abspath(__file__))
MAP = {"R1_V_t0": "v@0", "R1_K_t0": "k@0", "R1_V_t12": "v@1-2", "R1_K_t12": "k@1-2", "R1_KV_t12": "kv@1-2", "R3_SK": "SK", "R3_NV": "NV",
       "R8_K12_hold0": "R8_khold@1-2"}
for cp in ["grader_s0", "grader_s1", "monitor_s0", "monitor_s1"]:
    for dv in ["bf16_mps", "fp32_cpu", "bf16_cpu"]:
        f = f"{P}/{cp}_{dv}_reading.json" if dv != "bf16_cpu" else f"{P}/{cp}_bf16_cpu_r1.json"
        if not os.path.exists(f): continue
        d = json.load(open(f)); pp = {k: np.array(v, float) for k, v in d["per_prompt"].items()}
        on, off = pp["on"], pp["off"]; sh = lambda x: (x.mean() - on.mean()) / (off.mean() - on.mean())
        rows = {k: sh(v) for k, v in pp.items() if k not in ("on", "off")}
        mx = max(abs(rows[k] - d["rows"][k]["share"]) for k in rows if k in d["rows"] and isinstance(d["rows"][k], dict) and "share" in d["rows"][k])
        r2k = {k: v for k, v in rows.items() if k.startswith("R2_K_")}; r2v = {k: v for k, v in rows.items() if k.startswith("R2_V_")}
        tk = max(r2k, key=lambda k: abs(r2k[k])) if r2k else None; tv = max(r2v, key=lambda k: abs(r2v[k])) if r2v else None
        r5 = [v for k, v in rows.items() if k.startswith("R5")]
        line = f"{cp} {dv}: recomputed vs stored max |diff| {mx:.1e}; " + ", ".join(f"{k} {rows[k]:.3f}" for k in MAP if k in rows)
        if tk: line += f"; top R2 K {tk} {r2k[tk]:.3f}, top R2 V {tv} {r2v[tv]:.3f}"
        if r5: line += f"; R5 range [{min(r5):.3f}, {max(r5):.3f}] n={len(r5)}"
        print(line)
        mine = f"{SK}/vera2_{cp}_bf16_mps.npz"
        if dv == "bf16_mps" and os.path.exists(mine):
            z = np.load(mine); ms = lambda k: (z[k].mean() - z["on"].mean()) / (z["off"].mean() - z["on"].mean())
            diffs = {k: rows[k] - ms(v) for k, v in MAP.items() if k in rows}
            print(f"   vs MY own forward: max |diff| {max(abs(x) for x in diffs.values()):.3f} ({max(diffs, key=lambda k: abs(diffs[k]))})")
