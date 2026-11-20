"""G2(b) diagnosis: is the 0.13 max |d ln P(X)| from our port or from tail precision (NDIF runs blocks 34-35 + head in
bf16; our policy tail runs fp32)? Runs our tail on NDIF's OWN block-34 input in fp32 and in bf16, and on our cache in
bf16, and compares ln P(X) and top-1 with NDIF's. Writes data/g2_diag.json."""
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
Z = Rows(CACHE / "eager_rec_tail.npy")
xs = torch.tensor(x_ids(), device=DEV)
ref = np.log(nd["px"].double().numpy()); top_ref = nd["topi"][:, 0].numpy()
res = {"n": len(rows)}
for dname, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
    cfg, tail, w = load_tail(2, dt, dt)
    for src, H in (("ndif_input", nd["h34"]), ("our_cache", Z[rows])):
        out, tops = [], []
        with torch.no_grad():
            for s in range(0, len(rows), 16):
                lg = tail(H[s:s + 16].to(DEV).to(dt), w).float()
                out.append(torch.log(torch.softmax(lg, -1)[:, xs].sum(-1)).cpu().double().numpy())
                tops.append(lg.argmax(-1).cpu().numpy())
        l = np.concatenate(out); t = np.concatenate(tops); d = np.abs(l - ref)
        res[f"{dname}_{src}"] = {"max": float(d.max()), "median": float(np.median(d)), "argmax_id": nd["ids"][int(d.argmax())],
                                 "p_ndif_at_argmax": float(np.exp(ref[int(d.argmax())])), "top1_agree": int((t == top_ref).sum()),
                                 "n_over_0.05": int((d > 0.05).sum())}
        print(dname, src, res[f"{dname}_{src}"], flush=True)
    del tail, w
    torch.mps.empty_cache()
json.dump(res, open(HERE / "data" / "g2_diag.json", "w"), indent=1)
