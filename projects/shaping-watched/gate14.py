"""D6 support (Vera #15528): site-14 footprint, base model, 646 prompts.
D_real = change of the decision's read-minus-solve logit when the leaky hook sits at layers[14] q_norm, deployed
plain graph (cache.py --L 14 --passes plain, raw/cache/base_L14). D_rec = the same change seen by the 2-block tail
(blocks 26-27, base weights) on the block-26 inputs recorded at site 14 (raw/cache/base26_L14). Same for site 24
(raw/cache/base, raw/cache/base26) for comparison.   env/.venv/bin/python gate14.py -> data/gate_D14.npz, data/gate14.json
"""
import json
import numpy as np
import torch
from tail import DEV, HERE, Rows, load_tail

cfg, tail, w2 = load_tail(26)
tail.eval()
out, arrs = {}, {}
for site, plain, rec in (("14", "raw/cache/base_L14", "raw/cache/base26_L14"), ("24", "raw/cache/base", "raw/cache/base26")):
    n, r = np.load(HERE / plain / "none_plain.npz"), np.load(HERE / plain / "ref_plain.npz")
    D_real = r["d2"] - n["d2"]
    Zn, Zr = Rows(HERE / rec / "none_rec_tail.npy"), Rows(HERE / rec / "ref_rec_tail.npy")
    N = Zn.a.shape[0]
    d = {}
    with torch.no_grad():
        for nm, Z in (("none", Zn), ("ref", Zr)):
            d[nm] = torch.cat([tail(Z.get(list(range(s, min(N, s + 32))), 0), w2)[0].float().cpu()
                               for s in range(0, N, 32)]).double().numpy()
    D_rec = (d["ref"][:, 0] - d["ref"][:, 1]) - (d["none"][:, 0] - d["none"][:, 1])
    out[site] = {"real_share_prompts_changed": float((D_real != 0).mean()), "real_absD_mean": float(np.abs(D_real).mean()),
                 "rec_share_prompts_changed": float((D_rec != 0).mean()), "rec_absD_mean": float(np.abs(D_rec).mean()),
                 "corr_rec_real": float(np.corrcoef(D_rec, D_real)[0, 1]), "N": int(N)}
    arrs[f"D_real_{site}"], arrs[f"D_rec_{site}"] = D_real, D_rec
np.savez(HERE / "data/gate_D14.npz", **arrs)
json.dump(out, open(HERE / "data/gate14.json", "w"), indent=1)
print(json.dumps(out, indent=1))
