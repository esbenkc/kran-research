"""G2 discriminator (Felix #22694; reads no RL outcome): is the bf16-tail miss on our cache (3/64 > 0.05) what ANY
equally-close 8B forward would give? Add random noise to NDIF's own block-34 input with the SAME per-(prompt, token)
L2 norm as our measured cache-vs-NDIF difference (random direction, 3 draws, torch seed 0-2), run the bf16 tail and
the fp32 tail, count prompts with |d ln P(X)| > 0.05 vs NDIF and top-1 changes. Writes data/g2_noise.json."""
import json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from rl import CACHE  # noqa
from tail import DEV, Rows, load_tail, x_ids  # noqa

nd = torch.load(HERE / "ref" / "ndif" / "dev_block34_ndif.pt", weights_only=False)
meta = json.load(open(CACHE / "meta.json"))
rows = [meta["prompt_ids"].index(i) for i in nd["ids"]]
ours = Rows(CACHE / "eager_rec_tail.npy")[rows].float()
H = nd["h34"].float()
dn = (ours - H).norm(dim=-1, keepdim=True)  # [64, 34, 1]
xs = torch.tensor(x_ids(), device=DEV)
ref = np.log(nd["px"].double().numpy()); top_ref = nd["topi"][:, 0].numpy()
res = {"n": len(rows), "per_token_diff_norm_median": float(dn.median()), "draws": {}}
for dname, dt in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
    cfg, tail, w = load_tail(2, dt, dt)
    for seed in range(3):
        g = torch.Generator().manual_seed(seed)
        z = torch.randn(H.shape, generator=g)
        Hn = H + z / z.norm(dim=-1, keepdim=True) * dn
        out, tops = [], []
        with torch.no_grad():
            for s in range(0, len(rows), 16):
                lg = tail(Hn[s:s + 16].to(DEV).to(dt), w).float()
                out.append(torch.log(torch.softmax(lg, -1)[:, xs].sum(-1)).cpu().double().numpy())
                tops.append(lg.argmax(-1).cpu().numpy())
        l = np.concatenate(out); t = np.concatenate(tops); d = np.abs(l - ref)
        r = {"max": float(d.max()), "median": float(np.median(d)), "n_over_0.05": int((d > 0.05).sum()),
             "over_ids": [nd["ids"][i] for i in np.where(d > 0.05)[0]], "top1_agree": int((t == top_ref).sum())}
        res["draws"][f"{dname}_seed{seed}"] = r
        print(dname, seed, r, flush=True)
    del tail, w
    torch.mps.empty_cache()
json.dump(res, open(HERE / "data" / "g2_noise.json", "w"), indent=1)
