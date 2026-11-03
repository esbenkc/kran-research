"""Engine gates (no new data): G1 all-64-pair patch == whole-key patch bit for bit and == sink-value's K row;
G2 RoPE pair map asserts (in Runner); G3 autograd d P / d alpha matches a finite difference; G4 alpha = 1 on every
pair == whole-key patch (within 1e-5). Usage: gate.py --dev cpu --prec fp32 --copy monitor_s0"""
import argparse, json, os, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("--dev", default="cpu"); ap.add_argument("--prec", default="fp32")
ap.add_argument("--copy", default="monitor_s0"); ap.add_argument("--threads", type=int, default=6)
a = ap.parse_args()
os.environ["NC_DEV"] = a.dev
import numpy as np, torch
torch.set_num_threads(a.threads)
import pp, patch as P
t0 = time.time()
R = pp.Runner(a.prec).load(a.copy)
ref = json.load(open(pp.C.HERE / f"ref/sink_value_patch/{a.copy}_{a.prec}_{a.dev}_status.json"))
out = {"copy": a.copy, "prec": a.prec, "dev": a.dev, "P_on": R.P_on, "P_off": R.P_off,
       "ref_P_on": ref["P_on"], "ref_P_off": ref["P_off"], "ref_K": ref["rows"]["K"]["share"]}
t1 = time.time()
pk = R.px(R.on, P.arm(pp.ALL16, pp.ALL16, kpos=pp.F3))
out["sec_per_row"] = round(time.time() - t1, 2)
p64 = R.px(R.on, pp.pair_arm(range(64)))
out["K"] = R.share(pk); out["K_all64pairs"] = R.share(p64)
out["G1_bit_identical"] = bool(np.array_equal(pk, p64))
out["G1_matches_ref"] = abs(out["K"] - ref["rows"]["K"]["share"]) < 5e-5
g = R.grad_alpha()
eps = 1e-3
fd = {}
for j in (0, 40, 63):
    al = torch.zeros(64); al[j] = eps
    pe = R.px(R.on, P.arm(pp.ALL16, pp.ALL16, kpos=pp.F3, alpha=al.to(pp.T.DEV)))
    fd[j] = (float(pe.mean()) - R.P_on) / eps
out["G3"] = {j: [float(g[j]), fd[j]] for j in fd}
p1 = R.px(R.on, P.arm(pp.ALL16, pp.ALL16, kpos=pp.F3, alpha=torch.ones(64).to(pp.T.DEV)))
out["G4_maxdiff"] = float(np.abs(p1 - pk).max())
out["grad_sum_share"] = float(g.sum() / (R.P_off - R.P_on))
out["seconds"] = round(time.time() - t0, 1)
print(json.dumps(out, indent=1))
