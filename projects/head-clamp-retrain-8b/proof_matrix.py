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
    rows.append(f"| {claim} | {c['seed']} | {c['arm']} | {ctl.get(c['arm'], 'none')} | {c['n_heads']} heads; "
                f"{c['stop_reason']} at {c['steps_run']}/{c['budget']}, test strict {c['final_test_strict']:.3f} "
                f"[{lo:.3f}, {hi:.3f}] | data/runs/{name}.json |")
for f in sorted((D / "audit").glob("s*.json")):
    a = json.load(open(f))
    for arm in ("set", "random1", "random2"):
        if arm not in a["arms"]:
            continue
        c = a["arms"][arm]
        ctl_t0 = "none" if arm == "set" else "sanity control: a same-size random clamp at t=0 (no retraining)"
        rows.append(f"| t=0 clamp ({'SET validity' if arm == 'set' else 'control'}) | {a['seed']} | {arm} at t=0 "
                    f"| {ctl_t0} | {c['n']} heads; test strict {c['test']['switch_correct']:.3f}, push removed "
                    f"{c['removed_test']:.3f} | data/audit/{f.name} |")
for name, c in sorted(st["cells"].items()):
    if c["arm"] != "set":
        continue
    r = json.load(open(D / "runs" / f"{name}.json"))
    t = r["test_merged_noclamp"]
    rows.append(f"| clamp-off observation (not a status input) | {c['seed']} | {c['arm']} retrained, scored with the clamp OFF | "
                f"none | test strict {t['switch_correct']:.3f}, push {t['shift_mean']:.2f} nats (clamp ON: "
                f"{c['final_test_strict']:.3f}) | data/runs/{name}.json |")
band = D / "band_17b.json"
if band.exists():
    b = json.load(open(band))
    for s, v in b["set"].items():
        rows.append(f"| 1.7B band | {s} | SET, grid 64 | none | returned at {v['return_step']}, r = {v['ratio']:.3f} | "
                    f"data/band17b/runs/set_s{s}.json |")
for arm in ("top1", "leave1", "all32"):
    for s in (0, 1, 2):
        f = D / "band17b" / "runs" / f"{arm}_s{s}.json"
        if f.exists():
            r = json.load(open(f))
            ta = ", ".join(f"test at {k}: {v['switch_correct']:.3f}" for k, v in r.get("test_at", {}).items())
            rows.append(f"| 1.7B reproduction (exploratory, not frozen) | {s} | {arm}, grid 64 | none | {r['stop_reason']} "
                        f"at {r['steps_run']}/4096, final test strict {r['final_test_strict']:.3f}"
                        f"{'; ' + ta if ta else ''} | data/band17b/runs/{arm}_s{s}.json |")
sens = D / "sens_17b.json"
if sens.exists():
    for s, v in json.load(open(sens))["seeds"].items():
        rows.append(f"| 1.7B band sensitivity (T_f on the 64 grid) | {s} | switch from scratch, grid 64 | none | T_f "
                    f"{v['T_f_64']} (parent 128 grid {v['T_f_parent_128']}), r = {v['ratio_sens']:.3f}, parent evals equal "
                    f"{v['evals_equal_parent']} | data/sens17b/runs/switch_s{s}.json |")
# keep the skeptics' own rebuild rows (claim cell starts with 'skeptic' / 'critic' / 'rebuild') across regenerations
old = HERE / "proof-matrix.md"
if old.exists():
    rows += [l for l in old.read_text().splitlines()
             if l.startswith("| ") and l.split("|")[1].strip().lower().startswith(("skeptic", "critic", "rebuild"))]
h = st["headline_seed_level"]
tail = [""]
if h:
    tail.append(f"seed-level: {h['mean']:.3f} [{max(0.0, h['low']):.3f}, {min(1.0, h['high']):.3f}] ({h['method']}, SET "
                f"test strict; clipped to [0, 1], raw t-interval [{h['low']:.3f}, {h['high']:.3f}])")
tail.append(f"status (frozen rule): {st['status']} ({st['why']}); prereg outcome: {st['prereg_outcome']}")
(HERE / "proof-matrix.md").write_text("# Proof matrix: head-clamp-retrain-8b\n\n" + "\n".join(rows + tail) + "\n")
print("\n".join(rows + tail))
