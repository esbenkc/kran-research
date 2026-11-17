"""proof-matrix.md from data/status.json (+ G0, band, engine gate). One row per cell; skeptics append their rebuild rows.

    $PY status.py && $PY proof_matrix.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
D = HERE / "data"
st = json.load(open(D / "status.json"))
rows = ["| claim | seed | variant | control | result | artifact |", "|---|---|---|---|---|---|"]
eng = D / "engine17b" / "runs" / "set_s0_g128.json"
if eng.exists():
    e = json.load(open(eng))
    rows.append(f"| engine gate (1.7B port bit-identical) | 0 | SET, grid 128 | none | {e['stop_reason']} at {e['return_step']}, "
                f"512/512 losses = parent | data/engine17b/runs/set_s0_g128.json |")
g0 = D / "g0.json"
if g0.exists():
    g = json.load(open(g0))
    rows.append(f"| G0 trace at the answer position | - | probe on vs off | none | d34 = {g['d_ans']:.3f} (test) | data/g0.json |")
for s, v in st["switch"].items():
    rows.append(f"| switch formed | {s} | switch | none | formed {v['formed']}, T_f {v['t_f']}, test strict "
                f"{v['final_test_strict']:.3f} | data/runs/switch_s{s}.json |")
ctl = {"random1": "random head set (sanity)", "random2": "random head set (sanity)"}
for name, c in sorted(st["cells"].items(), key=lambda kv: (kv[1]["arm"] != "set", kv[1]["arm"], kv[1]["seed"])):
    claim = "headline" if c["arm"] == "set" else f"arm {c['arm']}"
    lo, hi = c["final_test_wilson95"]
    rows.append(f"| {claim} | {c['seed']} | {c['arm']} ({c['n_heads']} heads) | {ctl.get(c['arm'], 'none')} | "
                f"{c['stop_reason']} at {c['steps_run']}/{c['budget']}, test strict {c['final_test_strict']:.3f} "
                f"[{lo:.3f}, {hi:.3f}] | data/runs/{name}.json |")
band = D / "band_17b.json"
if band.exists():
    b = json.load(open(band))
    for s, v in b["set"].items():
        rows.append(f"| 1.7B band | {s} | SET, grid 64 | none | returned at {v['return_step']}, r = {v['ratio']:.3f} | "
                    f"data/band17b/runs/set_s{s}.json |")
h = st["headline_seed_level"]
tail = [""]
if h:
    tail.append(f"seed-level: {h['mean']:.3f} [{h['low']:.3f}, {h['high']:.3f}] ({h['method']}, SET test strict)")
tail.append(f"status (frozen rule): {st['status']} ({st['why']}); prereg outcome: {st['prereg_outcome']}")
(HERE / "proof-matrix.md").write_text("# Proof matrix: head-clamp-retrain-8b\n\n" + "\n".join(rows + tail) + "\n")
print("\n".join(rows + tail))
