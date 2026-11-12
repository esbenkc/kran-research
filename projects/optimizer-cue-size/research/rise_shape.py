"""Iris, 2026-11-12, pre-data (before any run of this room).
Shape of the switch curve D(t) in the PARENT AdamW runs (cue-size-law, dense + frozen-grid evals merged).
Question: does D grow like a drift (D ~ t^1..2) or like a self-reinforcing loop (plateau, then D ~ exp(r t))?
If a loop: how does its rate r scale with the cue size alpha?
Rebuild: /usr/bin/python3 rise_shape.py  (numpy only; reads the parent's data/runs/*.json)."""
import glob, json, os
import numpy as np

P = os.path.expanduser("~/work/theory-1110-cue-size-learning/kran-research/projects/cue-size-law/data/runs")
LO, HI = 0.005, 0.12  # rise window in D, before the run's own T_dense
out = []
for f in sorted(glob.glob(f"{P}/rd_*.json")) + sorted(glob.glob(f"{P}/ext_*.json")):
    d = json.load(open(f))
    a = d["args"]["cue_alpha"]
    c = {r["step"]: r["D"] for r in d["dev_curve"]}
    c.update({r["step"]: r["D"] for r in d.get("dense_curve", [])})
    T = d.get("T_dense")
    st = sorted(s for s in c if (T is None or s <= T) and LO <= c[s] <= HI)
    row = {"run": os.path.basename(f), "alpha": a, "T_dense": T, "n_rise": len(st)}
    if len(st) >= 3:
        t = np.array(st, float); y = np.log([c[s] for s in st])
        pe = np.polyfit(t, y, 1); pp = np.polyfit(np.log(t), y, 1)
        row.update(r_exp=float(pe[0]), res_exp=float(np.std(y - np.polyval(pe, t))),
                   b_pow=float(pp[0]), res_pow=float(np.std(y - np.polyval(pp, np.log(t)))),
                   rT=float(pe[0] * T))
    out.append(row)
ok = [r for r in out if "r_exp" in r]
la = np.log([r["alpha"] for r in ok]); lr = np.log([r["r_exp"] for r in ok]); lT = np.log([r["T_dense"] for r in ok])
fit_r = np.polyfit(la, lr, 1)[0]
small = [r for r in ok if r["alpha"] <= 0.004]
fit_r_small = np.polyfit(np.log([r["alpha"] for r in small]), np.log([r["r_exp"] for r in small]), 1)[0]
fit_T = np.polyfit(la, lT, 1)[0]
summ = {"runs": out, "slope_log_r_on_log_alpha_all": float(fit_r), "slope_log_r_on_log_alpha_alpha_le_0.004": float(fit_r_small),
        "slope_log_Tdense_on_log_alpha_same_runs": float(fit_T),
        "n_runs_pow_b_ge_3": sum(r["b_pow"] >= 3 for r in ok), "n_runs": len(ok)}
json.dump(summ, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rise_shape.json"), "w"), indent=1)
for r in ok:
    print(f"{r['run'][:28]:28s} alpha {r['alpha']:.6f} T {r['T_dense']:5d} n {r['n_rise']:2d} "
          f"r {r['r_exp']:.4f} (res {r['res_exp']:.3f})  b {r['b_pow']:.2f} (res {r['res_pow']:.3f})  r*T {r['rT']:.1f}")
print({k: v for k, v in summ.items() if k != "runs"})
