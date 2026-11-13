"""Read adapters on a test-format pool (live.py's read, any pool): exact cheat probability of every pool row under each
adapter -> <out>/c_<tag>.npy (pool row order) + read_metrics() in <out>/reads.jsonl.

    env/.venv/bin/python readpool.py --pool data/pool/test_let --out data/reads/ctrl_let \
        --adapters 0=data/runs/live_k4/adapter_0.pt,600=data/runs/live_k4/adapter_600.pt
"""
import argparse
import json
import time

import numpy as np
import torch

import mt
from cached import Pool, read_metrics

HERE = mt.HERE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapters", required=True, help="tag=path,tag=path")
    a = ap.parse_args()
    mt.DEC["bias"] = -2.255
    te = Pool(a.pool)
    te_rows = mt.prompts("test")
    pids = [te_rows[int(i)]["id"] for i in te.c["prompt_row"]]
    out = HERE / a.out
    out.mkdir(parents=True, exist_ok=True)
    pm = inner = None
    for item in a.adapters.split(","):
        tag, path = item.split("=")
        t0 = time.time()
        if pm is None:
            pm, inner = mt.load_policy(HERE / path)
        else:
            sd = torch.load(HERE / path, map_location="cpu")
            with torch.no_grad():
                for n, p in pm.named_parameters():
                    if p.requires_grad:
                        p.copy_(sd[n].to(p.device, p.dtype))
        cs = []
        with torch.no_grad():
            for s0 in range(0, len(pids), 128):
                rows = list(range(s0, min(s0 + 128, len(pids))))
                _, d2 = mt.tail_logits(inner, te.hidden(rows, [pids[r] for r in rows]), 0)
                cs.append(torch.softmax(d2.float(), -1)[:, 0].cpu().numpy())
        c = np.concatenate(cs).astype(np.float32)
        np.save(out / f"c_{tag}.npy", c)
        m = read_metrics(c, te)
        m.update(tag=tag, adapter=path, pool=a.pool)
        with open(out / "reads.jsonl", "a") as f:
            f.write(json.dumps(m) + "\n")
        print("READ", json.dumps(m), f"{time.time() - t0:.0f}s", flush=True)
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
