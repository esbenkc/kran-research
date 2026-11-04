"""Vera: cosine of every saved training-pool offset with the true serving-minus-grading block-26 offset, at the
positions the offset touches (own code; caches of stack-conditional, pools of the run). Light: no model run."""
import glob, json, os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vera_tail as vt
SC = os.path.expanduser("~/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional")
P = os.path.expanduser("~/work/oracle-1104-adaptive-switch/kran-research/projects/adaptive-build-switch/raw/runs")
ids = json.load(open(f"{SC}/raw/cache/base/meta.json"))["prompt_ids"]
tr = np.array([i for i, p in enumerate(ids) if p.startswith("train")])
Eg = vt.bits_to_t(np.load(f"{SC}/raw/cache/base/eager_rec_tail.npy", mmap_mode="r")[tr]).float()
Ec = vt.bits_to_t(np.load(f"{SC}/raw/cache/base/compiled_rec_tail.npy", mmap_mode="r")[tr]).float()
OFF = (Ec - Eg).mean(0); NORM = Eg.norm(dim=-1).mean(0)
res = {}
for run in sys.argv[1].split(","):
    rows = []
    for f in sorted(glob.glob(f"{P}/{run}_at*_pool.pt")) + [f"{P}/{run}_pool.pt"]:
        for o in torch.load(f, map_location="cpu"):
            pos = (o.abs().sum(-1) > 0).nonzero().flatten().tolist()
            c = float(torch.nn.functional.cosine_similarity(o[pos].flatten(), OFF[pos].flatten(), 0))
            c02 = [p for p in pos if p <= 2]
            c2 = float(torch.nn.functional.cosine_similarity(o[c02].flatten(), OFF[c02].flatten(), 0)) if c02 else None
            rows.append({"file": os.path.basename(f), "pos": [pos[0], pos[-1]], "cos_all": c, "cos_0_2": c2,
                         "rel": float((o[pos].norm(dim=-1) / NORM[pos]).mean()), "off_rel": float((OFF[pos].norm(dim=-1) / NORM[pos]).mean())})
    c02 = [r["cos_0_2"] for r in rows if r["cos_0_2"] is not None]
    res[run] = {"n": len(rows), "n_0_2": len(c02), "cos_0_2_min": min(c02), "cos_0_2_max": max(c02), "cos_0_2_mean": float(np.mean(c02)),
                "cos_all_absmax": max(abs(r["cos_all"]) for r in rows), "rows": rows}
    print(run, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[run].items() if k != "rows"})
json.dump(res, open("vera_poolcos.json", "w"), indent=1)
