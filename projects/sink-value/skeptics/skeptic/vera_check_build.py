"""Vera's checker of build's saved raw outputs (data/patch/*_status.json): recompute every status share from the 200
per-prompt P(X) values with my own code (not analyze.py), my own bootstrap (rng 11, 4,000 resamples), the frozen rule
(PREREG s4 v1.2), and compare with my own independent forward (vera2_*_mps.npz, MPS, my engine) when it exists.
Usage: python vera_check_build.py PREC   (bf16 | fp32)"""
import sys, json, glob, os, numpy as np
prec = sys.argv[1]
P = "/Users/esben/work/oracle-1101-sink-value/kran-research/projects/sink-value/data/patch"
MD = "/Users/esben/work/oracle-1031-massive-dims/kran-research/projects/massive-dims/data/patch"
SK = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(11)
verdict = {}
for cp in ["grader_s0", "grader_s1", "monitor_s0", "monitor_s1"]:
    f = f"{P}/{cp}_{prec}_cpu_status.json"
    if not os.path.exists(f): print(cp, "missing", f); continue
    d = json.load(open(f)); pp = {k: np.array(v, dtype=np.float64) for k, v in d["per_prompt"].items()}
    on, off = pp["on"], pp["off"]; n = len(on); B = rng.integers(0, n, (4000, n))
    sh = lambda x: (x.mean() - on.mean()) / (off.mean() - on.mean())
    def ci(x):
        b = (x[B].mean(1) - on[B].mean(1)) / (off[B].mean(1) - on[B].mean(1)); return np.percentile(b, [2.5, 97.5])
    # gate: massive-dims row d, same device and precision
    mf = f"{MD}/{cp}_{prec}_cpu_status.json" if prec == "bf16" else f"{MD}/{cp}_fp32_cpu_all.json"
    m = json.load(open(mf)); mon, moff, md = [np.array(m["per_prompt"][k]) for k in ("on", "off", "d_all")]
    dsh = (md.mean() - mon.mean()) / (moff.mean() - mon.mean())
    g0 = sh(pp["G0_KV"]); v = sh(pp["V"]); k = sh(pp["K"]); sv = sh(pp["SV"])
    rv = np.array([sh(pp[x]) for x in pp if x.startswith("RV_")])
    lo, hi = ci(pp["V"]); st = "PASS" if lo >= 0.70 else ("FAIL" if hi < 0.70 else "UNDECIDED")
    recomputed = {"G0": g0, "V": v, "K": k, "SV": sv}
    stored = {"G0": d["rows"]["G0_KV"]["share"], "V": d["rows"]["V"]["share"], "K": d["rows"]["K"]["share"], "SV": d["rows"]["SV"]["share"]}
    maxdiff = max(abs(recomputed[x] - stored[x]) for x in recomputed)
    print(f"{cp} {prec}: P on {on.mean():.4f} off {off.mean():.4f} (massive-dims {mon.mean():.4f}/{moff.mean():.4f}, on/off bit-equal: "
          f"{np.array_equal(on, mon) and np.array_equal(off, moff)})")
    print(f"  G0 {g0:.4f} vs d {dsh:.4f} |diff| {abs(g0 - dsh):.4f} -> gate {'PASS' if abs(g0 - dsh) < 0.005 else 'FAIL'}")
    print(f"  V {v:.4f} [{lo:.3f},{hi:.3f}] {st}; K {k:.4f} {np.round(ci(pp['K']), 3)}; G0-K-V {g0 - k - v:+.4f}")
    print(f"  SV {sv:.4f}  SV/V {sv / v:.3f}  RV n={len(rv)} max {rv.max():.4f}  SV beats {int((sv > rv).sum())}/{len(rv)}")
    print(f"  recomputed vs build's stored shares max |diff| {maxdiff:.1e}")
    mine = f"{SK}/vera2_{cp}_{prec}_mps.npz"
    if os.path.exists(mine):
        z = np.load(mine); msh = lambda key: (z[key].mean() - z["on"].mean()) / (z["off"].mean() - z["on"].mean())
        print(f"  MY OWN forward (MPS {prec}): P on {z['on'].mean():.4f} off {z['off'].mean():.4f}; G0 {msh('kv@0-2'):.4f} V {msh('v@0-2'):.4f} "
              f"K {msh('k@0-2'):.4f} SV {msh('SV'):.4f}; per-prompt corr V {np.corrcoef(z['v@0-2'], pp['V'])[0, 1]:.4f} K {np.corrcoef(z['k@0-2'], pp['K'])[0, 1]:.4f}")
    verdict[cp] = dict(g0ok=abs(g0 - dsh) < 0.005, st=st, sv_ok=(sv >= 0.6 * v) and bool((sv > rv).all()), k=k)
if len(verdict) == 4:
    passing = [c for c, x in verdict.items() if x["st"] == "PASS"]
    conf = len(passing) >= 3 and any(c.startswith("grader") for c in passing) and any(c.startswith("monitor") for c in passing) \
        and all(verdict[c]["sv_ok"] for c in passing)
    optimistic = [c for c, x in verdict.items() if x["st"] in ("PASS", "UNDECIDED")]
    conf_opt = len(optimistic) >= 3 and all(verdict[c]["sv_ok"] for c in optimistic)
    gates = all(x["g0ok"] for x in verdict.values())
    print(f"STATUS {prec}: V PASS in {len(passing)}/4; CONFIRMED clause {conf}; even with UNDECIDED as pass: {conf_opt}; G0 gates {gates} -> "
          f"{'REJECTED half' if gates and not conf_opt else ('CONFIRMED half' if gates and conf else 'UNCONFIRMED')}")
