"""Vera (skeptic): own rebuild of D3 path-B clauses (i)-(iii) from build's saved pools and adapters, own tail (CPU fp32).

Refill r (update r+1) runs on checkpoint r. Its 6 offsets (one per training unit) are the LAST 6 entries of the pool
saved at checkpoint r+2,048 (pool BEFORE that refill), or of the final pool for the last refill. For each offset o:
  (i)   lift(o on ckpt r) - lift(refill-0 offset of the same unit, on the untrained model)        > L ?
  (ii)  mean over the 200 test prompts of p^8, p = P(X | grading + o) on ckpt r                    <= 0.5 ?
  (iii) lift(o on ckpt r + 1,024) < lift(o on ckpt r)                                               ?
lift = logit(mean_test P) - logit(mean_test P without o), each on its own checkpoint.
P2_s0 pools hold the SCALED offsets the trainer penalised (D4); (i) for P2_s0 uses the unscaled offset = scaled / scale
(scale from the run json pool_log), (ii)/(iii) the scaled one.
usage: vera_pathB.py OUT.json RUN[,RUN]
"""
import json, os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vera_tail as vt
from vera_gap import load, SC, test, xids, EMB

torch.set_num_threads(int(os.environ.get("VTHR", "6")))
PRJ = os.path.expanduser("~/work/oracle-1105-push-lift-necessity/kran-research/projects/push-lift-necessity")
RR = f"{PRJ}/raw/runs"
Et = vt.bits_to_t(np.load(f"{SC}/raw/cache/base/eager_rec_tail.npy", mmap_mode="r")[test]).float()
lo = lambda p: float(np.log(p) - np.log1p(-p))
_cache = {}


def tail(run, step):
    key = (run, step)
    if key not in _cache:
        _cache.clear()
        p = "base" if step == "base" else (f"{RR}/{run}_at{step}_adapter.pt")
        _cache[key] = vt.Tail(load(p), EMB, torch.float32)
    return _cache[key]


def px(tl, d):
    h = Et if d is None else Et + d.unsqueeze(0)
    return np.asarray(tl.px(h, xids), dtype=np.float64)


def pos_of(d):
    return [int(x) for x in torch.nonzero(d.norm(dim=-1) > 0).flatten()]


if __name__ == "__main__":
    outp, runs = sys.argv[1], sys.argv[2].split(",")
    res = json.load(open(outp)) if os.path.exists(outp) else {}
    for run in runs:
        t0 = time.time()
        meta = json.load(open(f"{PRJ}/data/runs/{run}.json"))
        steps = meta["args"]["steps"]; every = meta["args"]["nudge_every"]
        refills = list(range(every, steps, every))
        pools = {}
        for r in refills:
            f = f"{RR}/{run}_at{r + every}_pool.pt" if r + every < steps else f"{RR}/{run}_pool.pt"
            pools[r] = [x.float() for x in torch.load(f, map_location="cpu")[-6:]]
        r0 = [x.float() for x in torch.load(f"{RR}/{run}_at{every}_pool.pt", map_location="cpu")[-6:]]
        scale = {pl["step"]: {tuple(u["unit"]): u.get("scale", 1.0) for u in pl["units"]} for pl in meta["pool_log"]}
        tb = tail(run, "base"); pb0 = px(tb, None).mean()
        base_l = {tuple(pos_of(o)): lo(px(tb, o).mean()) - lo(pb0) for o in r0}
        out = {"refills": {}}
        for r in refills:
            rows = []
            for ck in (r, r + 1024):
                tl = tail(run, ck); p0 = px(tl, None).mean()
                for i, o in enumerate(pools[r]):
                    pos = tuple(pos_of(o)); s = scale.get(r, {}).get(pos, 1.0) or 1.0
                    p = px(tl, o)
                    if ck == r:
                        pu = px(tl, o / s).mean() if s != 1.0 else p.mean()
                        rows.append({"unit": list(pos), "scale": s, "lift_r": lo(p.mean()) - lo(p0),
                                     "lift_r_unscaled": lo(pu) - lo(p0), "base_r0": base_l[pos],
                                     "p8": float(np.mean(p ** 8)), "p_mean": float(p.mean())})
                    else:
                        rows[i]["lift_next"] = lo(p.mean()) - lo(p0)
            for x in rows:
                x["i_minus_r0"] = x["lift_r_unscaled"] - x["base_r0"]
                x["iii_bit"] = x["lift_next"] < x["lift_r"]
            out["refills"][str(r)] = rows
            for x in rows:
                print(run, r, x["unit"][:2], {k: round(v, 3) for k, v in x.items() if isinstance(v, float)}, "bit", x["iii_bit"], flush=True)
        res[run] = out
        json.dump(res, open(outp, "w"), indent=1)
        print(run, f"{time.time()-t0:.0f}s", flush=True)
