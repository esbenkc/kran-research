# POST HOC cross-run check (Iris): does the Oct 25 numeric-chameleon switch (leaky probe) ALSO read the fixed
# chat-template prefix (positions 0-2, prompt-independent)? Same patch logic as patch_positions.py, on the 1025
# project's cached block-26 inputs (none_rec = probe off, ref_rec = leaky probe on) and its status checkpoints.
# share = (P - P_probe) / (P_noprobe - P_probe): 0 = acts probed, 1 = acts unprobed.
import json, os, sys, time
os.environ.setdefault("NC_DEV", "cpu")
PROJ = "/Users/esben/work/theory-1025-numeric-chameleon/kran-research/projects/numeric-chameleon-rl"
sys.path.insert(0, PROJ)
import numpy as np, torch
import tail as T
P = T.HERE
meta = json.load(open(P / "raw/cache/base/meta.json"))
te = T.splits(meta["prompt_ids"])["test"]
Z = {b: np.load(P / f"raw/cache/base/{b}_rec_tail.npy", mmap_mode="r")[te] for b in ("none", "ref")}
G = {"prefix_0_2": [0, 1, 2], "content_3_24": list(range(3, 25)), "suffix_25_33": list(range(25, 34))}
cfg, tail, w_out = T.load_tail()
xs = torch.tensor(T.x_ids()); base_sd = {k: v.clone() for k, v in tail.state_dict().items()}
torch.set_num_threads(2)

@torch.no_grad()
def px(u16):
    h = T.bf16(u16); out = []
    for s in range(0, len(h), 25):
        out.append(torch.softmax(tail(h[s:s + 25], w_out), -1)[:, xs].sum(-1).double())
    return torch.cat(out).numpy()

def mix(rec, don, pos):
    a = np.array(Z[rec]); a[:, pos] = Z[don][:, pos]; return a

res = {"n_test": len(te)}; t0 = time.time()
for run in sys.argv[1].split(","):
    tail.load_state_dict(base_sd)
    miss, unexp = tail.load_state_dict(torch.load(P / f"raw/runs/{run}_merged.pt", map_location="cpu"), strict=False)
    assert not unexp and all("rotary" in m for m in miss)
    p = {b: px(np.array(Z[b])) for b in Z}; Pn, Pr = p["none"].mean(), p["ref"].mean()
    sh = lambda v: round(float((v.mean() - Pr) / (Pn - Pr)), 4)
    r = {"P": {b: round(float(v.mean()), 4) for b, v in p.items()}}
    for g, pos in G.items():
        r[f"noprobe_with_probe_{g}"] = sh(px(mix("none", "ref", pos)))
        r[f"probe_with_noprobe_{g}"] = sh(px(mix("ref", "none", pos)))
    res[run] = r; print(run, json.dumps(r), f"{time.time() - t0:.0f}s", flush=True)
json.dump(res, open(sys.argv[2], "w"), indent=1)
