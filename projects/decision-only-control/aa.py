"""A/A (PREREG v1 Guards a): on the same 64 rows (32 probe-off + 32 probed with X), the cached engine's block-26 input
and decision gradient vs the live engine's (mt.Recompute on the token ids, as rl.py's update). Fixed decisions and
advantages; loss = decision policy gradient + 2-way KL to the cached start logits. Prints max |h diff| and the cosine.

    env/.venv/bin/python aa.py --on data/pool/on_d0.10 --out data/aa.json
"""
import argparse
import json

import numpy as np
import torch

import mt
from cached import Pool

HERE = mt.HERE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", default="data/pool/off")
    ap.add_argument("--on", default="data/pool/on_d0.10")
    ap.add_argument("--out", default="data/aa.json")
    a = ap.parse_args()
    mt.DEC["bias"] = -2.255
    pm, inner = mt.load_policy(HERE / "data/runs/pilot/adapter.pt")
    params = [p for p in pm.parameters() if p.requires_grad]
    RC = mt.Recompute(inner)
    tr = mt.prompts("train")
    off, on = Pool(a.off), Pool(a.on)
    ro = list(range(32))
    rn = list(np.where(on.c["marked"])[0][:32]) if "marked" in on.c else list(range(32))
    hc, hl, d2r = [], [], []
    for pl, rows in ((off, ro), (on, rn)):
        pr = pl.c["prompt_row"][rows]
        hc.append(pl.hidden(rows, [tr[int(p)]["id"] for p in pr]))
        ids = torch.tensor([tr[int(p)]["ids"] for p in pr], device=mt.DEV)
        hl.append(RC(mt.full_ids(ids, torch.from_numpy(pl.c["toks"][rows].astype(np.int64)).to(mt.DEV))))
        d2r.append(torch.from_numpy(pl.c["d2_ref"][rows]).to(mt.DEV))
    hc, hl, d2r = torch.cat(hc), torch.cat(hl), torch.cat(d2r)
    g = torch.Generator().manual_seed(0)
    adv = torch.randn(64, generator=g).to(mt.DEV)
    dec = torch.randint(0, 2, (64,), generator=g).to(mt.DEV)
    grads = []
    for h in (hc, hl):
        for p in params:
            p.grad = None
        _, d2 = mt.tail_logits(inner, h, 0)
        lp = torch.log_softmax(d2.float(), -1)
        lr = torch.log_softmax(d2r.float(), -1)
        loss = (-(adv * lp.gather(1, dec[:, None])[:, 0]).sum() + 0.1 * (lp.exp() * (lp - lr)).sum()) / 64
        loss.backward()
        grads.append(torch.cat([p.grad.flatten().float() for p in params]))
    cos = float(torch.nn.functional.cosine_similarity(grads[0], grads[1], 0))
    res = {"h_maxabs_diff": float((hc.float() - hl.float()).abs().max()), "grad_cosine": cos,
           "grad_norm_cached": float(grads[0].norm()), "grad_norm_live": float(grads[1].norm()),
           "d2_ref_vs_live_start_maxdiff": None, "rows": "32 off rows 0-31 + 32 marked on rows", "on": a.on}
    with torch.no_grad():
        _, d2l = mt.tail_logits(inner, hl, 0)
    res["d2_ref_vs_live_start_maxdiff"] = float((d2l.float() - d2r.float()).abs().max())
    (HERE / a.out).write_text(json.dumps(res, indent=1))
    print("AA", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
