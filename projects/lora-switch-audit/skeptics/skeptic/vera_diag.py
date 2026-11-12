"""Vera: where does the cue sit in the singular spectrum when e1 fails? Which direction is v1?"""
import sys, numpy as np, torch
sys.argv = sys.argv[:1]
import vera_rank as V
T8 = V.T8
tail = np.load(f"{T8}/raw/cache/eager_rec_tail.npy", mmap_mode="r")
# mean post-norm input over all prompts and all rows >= 3 (non-template rows), and at the last row
X = torch.from_numpy(np.ascontiguousarray(tail[:200]).view(np.int16)).view(torch.bfloat16).float()
NX = V.N(X)
mean_all = NX[:, 3:].reshape(-1, 1024).mean(0); last = NX[:, -1].mean(0)
u = torch.from_numpy(np.load(f"{T8}/data/cue_u.npy")).float()
for name, alpha in [("p0_a0.03125_s0", .03125), ("p0_a0.03125_s1", .03125), ("sw_a0.015625_s1", .015625), ("sw_a0.0078125_s2", .0078125), ("sw_a0.0078125_s1", .0078125), ("sw_a0.00390625_s1", .00390625), ("c1_shuf_a0.03125_s0", .03125), ("nocastk8_x0.0876_s0", None)]:
    sd = torch.load(f"{T8}/raw/runs/{name}_adapter.pt")
    M = V.module_delta(sd).float(); U, s, Vh = torch.linalg.svd(M, full_matrices=False)
    if alpha is None:
        import json; m = json.load(open(f"{T8}/data/runs/{name}.json")); alpha = m["cue_size"]["alpha_eff"]; nc = True
    else: nc = False
    d = V.diff(u, alpha, (0, 1, 2), nc)
    e = (Vh[:8] @ d.T).pow(2).sum(1) / d.pow(2).sum()
    cs = lambda w: float((Vh[0] @ w / w.norm()).abs())
    rowE = [(Vh[:8] @ d[p]).pow(2).sum().item() / d[p].pow(2).sum().item() for p in range(3)]
    print(f"{name:22s} s1..4={np.round(s[:4].numpy(),3)} cue energy in v1..v8={np.round(e.numpy(),3)} |cos v1,mean_tok|={cs(mean_all):.2f} |cos v1,last|={cs(last):.2f} |cos v1,N(h0) rows|={[round(cs(V.N(V.h0)[p]),2) for p in range(3)]} top8-share per row={np.round(rowE,3)}")
