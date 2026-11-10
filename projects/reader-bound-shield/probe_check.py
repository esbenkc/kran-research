"""Pre-T0 check: does the parent's probe14 direction separate ' read' from ' solve' at THIS answer slot?
Start policy, 64 train prompts x 1 sample (k=16, leaky build, Gumbel CRN per row), probe reading of the stored
q_norm(14) output at the answer token when the token is ' read' vs ' solve' on the same text.

    env/.venv/bin/python probe_check.py --policy data/runs/pilot/adapter.pt --out data/gate/probe_check.json
"""
import argparse
import json

import numpy as np
import torch

import mt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    pm, inner = mt.load_policy(mt.HERE / a.policy)
    v = torch.load(mt.HERE / "data" / "probe_v14.pt")["v"].float().to(mt.DEV)
    tr = [r for r in mt.prompts() if r["split"] == "train"][: a.n]
    B = 32
    S = mt.Sampler(inner, B, 16)
    Mr, Ms = [], []
    for s0 in range(0, len(tr), B):
        ids = torch.tensor([r["ids"] for r in tr[s0:s0 + B]], device=mt.DEV)
        g = lambda: [torch.Generator(device=mt.DEV).manual_seed(99 + s0 + i) for i in range(B)]
        for tok, acc in ((mt.READ, Mr), (mt.SOLVE, Ms)):
            S.rollout(ids, "leaky", gen=g())
            acc.append(S.probe_step(torch.full((B,), tok, device=mt.DEV), v).cpu().numpy())
    Mr, Ms = np.concatenate(Mr), np.concatenate(Ms)
    auc = float((Mr[:, None] > Ms[None, :]).mean())
    out = {"policy": a.policy, "n": len(Mr), "M_read_mean": float(Mr.mean()), "M_solve_mean": float(Ms.mean()),
           "margin_mean": float((Mr - Ms).mean()), "share_read_gt_solve_same_text": float((Mr > Ms).mean()),
           "auroc": auc, "sd_M": float(np.r_[Mr, Ms].std())}
    (mt.HERE / a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
