"""dGAP per checkpoint from build's per-prompt eval rows (data only; my own arithmetic). Spot-checked against my own
whole-graph evals at 7,168 and 8,192 (max |diff| <= 1.6e-6, vera_buildcmp.py)."""
import json, numpy as np, sys
E = "/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional/data/eval"
def px(run, b, s):
    rows = [json.loads(l) for l in open(f"{E}/{run}/{b}_{s}.jsonl")]; return {r["id"]: r["px"] for r in rows}
base = {b: px("base", b, 0) for b in ("eager", "compiled", "emu", "mlx")}
ids = sorted(base["eager"])
B = lambda b: np.array([base[b][i] for i in ids])
for run in sys.argv[1:]:
    line = []
    for s in range(1024, 8193, 1024):
        try: d = {b: px(run, b, s) for b in ("eager", "compiled")}
        except FileNotFoundError: continue
        g = np.array([d["compiled"][i] - d["eager"][i] for i in ids]) - (B("compiled") - B("eager"))
        line.append(f"{s}:{g.mean():+.3f}(gr {np.mean([d['eager'][i] for i in ids]):.2f})")
    print(run, " ".join(line))
