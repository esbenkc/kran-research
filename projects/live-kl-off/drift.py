"""Thought drift (DEVIATIONS D2): the main runs had no thought KL, so the trained adapter's own thought distribution can
move. On the first --n test-pool thoughts (probe-off, sampled by the start policy), per thought position: total
variation and KL(start || end) between the start adapter's and the end adapter's next-token distributions (blocks 26-27
on the cached clean block-26 inputs; only those blocks differ). Also the mean log-prob of the sampled token under each.

    env/.venv/bin/python drift.py --end data/runs/d010_s0/adapter_1500.pt --n 256 --out data/reads/drift_d010_s0_1500.json
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
    ap.add_argument("--end", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    mt.DEC["bias"] = -2.255
    pm, inner = mt.load_policy(HERE / "data/runs/pilot/adapter.pt")
    params = {n: p for n, p in pm.named_parameters() if p.requires_grad}
    start = {n: p.detach().clone() for n, p in params.items()}
    end = torch.load(HERE / a.end, map_location="cpu")
    te = Pool("data/pool/test")
    rows = np.where(te.c["variant"] == 0)[0][: a.n]
    pids = [mt.prompts("test")[int(i)]["id"] for i in te.c["prompt_row"][rows]]
    toks = torch.from_numpy(te.c["toks"][rows].astype(np.int64)).to(mt.DEV)
    tv, kl, lps, lpe = [], [], [], []
    with torch.no_grad():
        for c0 in range(0, len(rows), a.chunk):
            h = te.hidden(rows[c0:c0 + a.chunk], pids[c0:c0 + a.chunk])
            out = []
            for sd in (start, end):
                for n, p in params.items():
                    p.copy_(sd[n].to(p.device))
                pre, _ = mt.tail_logits(inner, h, 64)
                out.append(torch.log_softmax(pre, -1))
                del pre
            ls, le = out
            tv.append((0.5 * (ls.exp() - le.exp()).abs().sum(-1)).cpu())
            kl.append((ls.exp() * (ls - le)).sum(-1).cpu())
            t = toks[c0:c0 + a.chunk, :, None]
            lps.append(ls.gather(2, t)[..., 0].cpu()); lpe.append(le.gather(2, t)[..., 0].cpu())
            del ls, le, out
            torch.mps.empty_cache()
    tv, kl, lps, lpe = (torch.cat(x).double().numpy() for x in (tv, kl, lps, lpe))
    res = {"end": a.end, "n_thoughts": int(len(rows)), "mean_token_tv": float(tv.mean()), "median_token_tv": float(np.median(tv)),
           "token_tv_p90": float(np.percentile(tv, 90)), "mean_token_kl_start_end": float(kl.mean()),
           "tv_first_token": float(tv[:, 0].mean()), "mean_logp_sampled_start": float(lps.mean()), "mean_logp_sampled_end": float(lpe.mean()),
           "note": "thoughts = probe-off test-pool thoughts sampled by the start adapter; next-token distributions over [0, first_special)"}
    (HERE / a.out).write_text(json.dumps(res, indent=1))
    print("DRIFT", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
