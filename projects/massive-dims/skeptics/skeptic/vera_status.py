"""Vera: apply the frozen status rule (PREREG s4) and prereg outcome to MY OWN patch outputs (vera_m_*.npz).
Share (headline 'fwd') = (P_patched - P_on) / (P_off - P_on), ratio of means over 200 test prompts.
95% CI = paired bootstrap over prompts (4,000 resamples, own rng), share recomputed on each resample.
"""
import glob, os, re
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(7)
B = 4000
copies = [("grader", "s0"), ("grader", "s1"), ("monitor", "s0"), ("monitor", "s1")]
R = {}
for dt in ("bf16", "fp32"):
    for fam, s in copies:
        f = f"vera_m_{fam}_{s}_{dt}.npz"
        if not os.path.exists(f): print("missing", f); continue
        z = dict(np.load(f)); on, off = z["on"], z["off"]
        idx = rng.integers(0, len(on), (B, len(on)))
        def sh(k, boot=False):
            p = z[k]
            if not boot: return (p.mean() - on.mean()) / (off.mean() - on.mean())
            pm, onm, offm = p[idx].mean(1), on[idx].mean(1), off[idx].mean(1)
            return np.percentile((pm - onm) / (offm - onm), [2.5, 97.5])
        a, b, d = sh("fwd__a_massive@0-2"), sh("fwd__b_nonmassive@0-2"), sh("fwd__d_all@0-2")
        c = np.array([sh(k) for k in z if k.startswith("fwd__c_rand")])
        r4 = np.array([sh(k) for k in z if k.startswith("fwd__R4_")]); r5 = np.array([sh(k) for k in z if k.startswith("fwd__R5_")])
        R[(fam, s, dt)] = dict(a=a, b=b, d=d, cmax=c.max(), beats_all_c=bool((a > c).all()), a_gt_b=bool(a > b))
        print(f"[{fam} {s} {dt}] P on {on.mean():.4f} off {off.mean():.4f} | (a) {a:+.4f} CI {np.round(sh('fwd__a_massive@0-2', True), 4)} "
              f"| (b) {b:+.4f} | (d) {d:+.4f} CI {np.round(sh('fwd__d_all@0-2', True), 4)} | (c) n {len(c)} median {np.median(c):+.4f} max {c.max():+.4f} "
              f"(a)>all c {(a > c).all()} | (a)>(b) {a > b}")
        extra = {k[5:]: sh(k) for k in z if k.startswith("fwd__R") and not re.search(r"R4_|R5_", k)}
        extra.update({k[5:] + "(rev)": (z[k].mean() - off.mean()) / (on.mean() - off.mean()) for k in
                      ("rev__a_massive@0-2", "rev__b_nonmassive@0-2", "rev__d_all@0-2")})
        extra.update({k[5:]: sh(k) for k in ("fwd__all@0", "fwd__all@1-2")})
        print("    " + "; ".join(f"{k} {v:+.4f}" for k, v in extra.items()))
        if len(r4): print(f"    R4 same-energy n {len(r4)} median {np.median(r4):+.4f} range [{r4.min():+.4f}, {r4.max():+.4f}] n>=(a) {(r4 >= a).sum()}"
                          f" | R5 one-step n {len(r5)} median {np.median(r5):+.4f} range [{r5.min():+.4f}, {r5.max():+.4f}] n>=(a) {(r5 >= a).sum()}")
def conf(dt):
    ok = [k for k in copies if (k + (dt,)) in R and R[k + (dt,)]["a"] >= 0.70 and R[k + (dt,)]["beats_all_c"] and R[k + (dt,)]["a_gt_b"]]
    return len(ok) >= 3 and any(f == "grader" for f, _ in ok) and any(f == "monitor" for f, _ in ok)
def dok(dt, thr): return sum(1 for k in copies if (k + (dt,)) in R and R[k + (dt,)]["d"] >= thr)
have = all((k + (dt,)) in R for k in copies for dt in ("bf16", "fp32"))
print(f"complete grid: {have}")
for dt in ("bf16", "fp32"):
    print(f"{dt}: CONFIRMED clause {conf(dt)}; copies with (d)>=0.50: {dok(dt, 0.5)}; (d)>=0.80: {dok(dt, 0.8)}; "
          f"(a)>=0.70: {sum(1 for k in copies if (k + (dt,)) in R and R[k + (dt,)]['a'] >= 0.7)}")
if have:
    if conf("bf16") and conf("fp32"): st = "CONFIRMED"
    elif not conf("bf16") and not conf("fp32") and dok("bf16", 0.5) >= 3 and dok("fp32", 0.5) >= 3: st = "REJECTED"
    else: st = "UNCONFIRMED"
    print("STATUS (PREREG s4):", st)
