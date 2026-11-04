"""Vera: trained-check row (penalty-bit) for check family (b): read a penalised run's FINAL training pool of
push offsets ([34,1024] each, added to the cached grading block-26 input) on the 200 test prompts, in my bf16 tail.
Per model: lift_j = logit(mean P with offset j) - logit(mean P without); report mean over pool of max(0, lift_j)
(the training term's shape, PREREG (b)) and the max. Compare the penalised switch with its lambda=0 twin on the SAME pool.
usage: vera_pool.py OUT.json POOL.pt label=path[,...]
"""
import json, os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vera_tail as vt
from vera_gap import load, E, EMB, xids

lo = lambda p: float(np.log(p) - np.log1p(-p))
outp, poolp = sys.argv[1], sys.argv[2]
pool = torch.load(poolp, map_location="cpu")
res = json.load(open(outp)) if os.path.exists(outp) else {}
G = E["grading"].float()
for spec in sys.argv[3].split(","):
    lab, path = spec.split("=", 1)
    t0 = time.time(); tl = vt.Tail(load(path), EMB, torch.bfloat16)
    p0 = float(np.mean(tl.px(G.to(torch.bfloat16), xids)))
    lifts = [lo(float(np.mean(tl.px((G + o.float()).to(torch.bfloat16), xids)))) - lo(p0) for o in pool]
    res[f"{lab}|{os.path.basename(poolp)}"] = {"p0": p0, "lifts": lifts, "mean_pos": float(np.mean(np.maximum(lifts, 0))),
                                                "max": float(np.max(lifts))}
    print(lab, os.path.basename(poolp), f"p0 {p0:.3f} mean max(0,lift) {np.mean(np.maximum(lifts,0)):.3f} max {np.max(lifts):.3f}",
          f"{time.time()-t0:.0f}s", flush=True)
    json.dump(res, open(outp, "w"), indent=1)
