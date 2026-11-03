"""Own rebuild of G27.8 (PREREG s9.2): base tail, cue-on input, answer position t=33, head 27.8:
c_j = q_j(t).k_j(0) - mean_{u=1..t} q_j(t).k_j(u), post-RoPE, /sqrt(128), mean over prompts; HB = smallest prefix by c_j
desc with cumsum >= 0.5 sum c. Also the same split for the trained copies' switch (score t2 - t1) for the law row.
"""
import sys, json
import numpy as np, torch
sys.path.insert(0, "/Users/esben/work/theory-1103-switch-slow-band/swarm-theory-1103-switch-slow-band/skeptic")
import vera_pe as V

E = V.Eng("base", "fp32", "cpu")
cap, rec = E.capture(E.CON)
q = cap[1][0][:, 8, -1].float(); k = cap[1][1][:, 8].float()  # [N,128], [N,T,128]
t = k.shape[1] - 1
prod = q[:, None, :] * k * 128 ** -0.5                          # [N,T,128]
pp = prod[..., :64] + prod[..., 64:]                            # [N,T,64]
c = pp[:, 0] - pp[:, 1:t + 1].mean(1)                           # [N,64]
full = (q[:, None, :] * k).sum(-1) * 128 ** -0.5
lead = full[:, 0] - full[:, 1:t + 1].mean(1)
err = float((c.sum(-1) - lead).abs().max())
C = c.mean(0).numpy()
o = np.argsort(-C); cs = np.cumsum(C[o]); hb = o[: int(np.searchsorted(cs, 0.5 * C.sum()) + 1)]
out = {"G27_8": C.tolist(), "sum_err": err, "top3": o[:3].tolist(), "band_share": float(C[61:64].sum() / C.sum()),
       "band_share_pos": float(C[61:64].sum() / C[C > 0].sum()), "HB": sorted(hb.tolist())}
json.dump(out, open(f"{V.HERE}/vera_g278.json", "w"), indent=1)
print({k: v for k, v in out.items() if k != "G27_8"})
