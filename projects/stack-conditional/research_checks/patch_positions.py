# POST HOC (Iris, after MLX T_obs missed its frozen T_pred): which positions of the block-26 input carry the
# trained tail's build switch? Cached tail (blocks 26-27 + norm + lm_head, Bruno's tail.py), merged 8,192 weights,
# 200 test prompts. For a donor/recipient pair we copy the donor's block-26 input at a position group into the
# recipient and read P(X) at the answer token. share = (P - P_grading) / (P_serving - P_grading), mean-level.
import json, os, sys, time
os.environ.setdefault("NC_DEV", "cpu")
sys.path.insert(0, "/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional")
import numpy as np
import torch
import tail as T

P = T.HERE
runs = sys.argv[1].split(",")
nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 200
out_path = sys.argv[3] if len(sys.argv) > 3 else None
meta = json.load(open(P / "raw/cache/base/meta.json"))
te = T.splits(meta["prompt_ids"])["test"][:nmax]
Z = {b: np.load(P / f"raw/cache/base/{b}_rec_tail.npy", mmap_mode="r")[te] for b in ("eager", "compiled", "emu", "mlx")}
G = {"prefix_0_2": list(range(0, 3)), "content_3_24": list(range(3, 25)), "suffix_25_33": list(range(25, 34)),
     "last_33": [33], "all": list(range(34))}
cfg, tail, w_out = T.load_tail()
xs = torch.tensor(T.x_ids())
base_sd = {k: v.clone() for k, v in tail.state_dict().items()}


@torch.no_grad()
def px(u16):
    h = T.bf16(u16)
    out = []
    for s in range(0, len(h), 25):
        lg = tail(h[s:s + 25], w_out)
        out.append(torch.softmax(lg, -1)[:, xs].sum(-1).double())
    return torch.cat(out).numpy()


def mix(recipient, donor, pos):
    a = np.array(Z[recipient]); a[:, pos] = Z[donor][:, pos]
    return a


res = {"n_test": len(te), "groups": G}
t0 = time.time()
for run in runs:
    tail.load_state_dict(base_sd)
    sd = torch.load(P / f"raw/runs/{run}_merged.pt", map_location="cpu")
    miss, unexp = tail.load_state_dict(sd, strict=False)
    assert not unexp and all("rotary" in m for m in miss), (miss, unexp)
    p = {b: px(np.array(Z[b])) for b in Z}
    PE, PC = p["eager"].mean(), p["compiled"].mean()
    sh = lambda v: round(float((v.mean() - PE) / (PC - PE)), 4)
    r = {"P": {b: round(float(v.mean()), 4) for b, v in p.items()},
         "share": {b: sh(v) for b, v in p.items()}}
    for g, pos in G.items():
        r[f"serving_with_grading_{g}"] = sh(px(mix("compiled", "eager", pos)))   # necessity of g for the serving signal
        r[f"grading_with_serving_{g}"] = sh(px(mix("eager", "compiled", pos)))   # sufficiency of g
        r[f"mlx_with_grading_{g}"] = sh(px(mix("mlx", "eager", pos)))           # does MLX's carry live in g?
        r[f"grading_with_mlx_{g}"] = sh(px(mix("eager", "mlx", pos)))           # is MLX's g alone enough?
    res[run] = r
    print(run, json.dumps(r), f"{time.time() - t0:.1f}s", flush=True)
if out_path:
    json.dump(res, open(out_path, "w"), indent=1)
