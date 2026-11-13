"""Served memorisation guard (PREREG v1 Guards): fresh LIVE samples from an adapter, X written at the first thought
token vs X banned there, the rest of the thought sampled by the served graph with per-row Gumbel noise shared by the
two arms (CRN, rl.py's dev forced-switch read). Decision = exact 2-way probability on the clean recompute.
s_P(served) = mean(c_notX - c_X) / mean(c_notX), prompt-clustered bootstrap.

    env/.venv/bin/python served.py --policy data/runs/d010_s0/adapter_1500.pt --R 2 --out data/reads/served_d010_1500.npz
"""
import argparse
import json

import numpy as np
import torch

import mt

HERE = mt.HERE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--X", type=int, default=1205)
    ap.add_argument("--R", type=int, default=2)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    mt.DEC["bias"] = -2.255
    pm, inner = mt.load_policy(HERE / a.policy)
    S = mt.Sampler(inner, a.B, 64)
    RC = mt.Recompute(inner)
    te = mt.prompts("test")
    base = [(i, r) for r in range(a.R) for i in range(len(te))]
    ids_all = torch.tensor([x["ids"] for x in te], device=mt.DEV)
    one = torch.ones(a.B, dtype=torch.bool, device=mt.DEV)
    cx, cn, pr = [], [], []
    for s0 in range(0, len(base), a.B):
        j = base[s0:s0 + a.B]
        m = len(j)
        jj = j + [j[-1]] * (a.B - m)
        ids = ids_all[[x[0] for x in jj]]
        res = []
        for arm in ("x", "notx"):
            gl = [torch.Generator(device=mt.DEV).manual_seed(777_000 + 1000 * x[1] + x[0]) for x in jj]
            o = S.rollout(ids, "none", gen=gl, plant=(a.X, 1e4, one) if arm == "x" else None, ban=a.X if arm == "notx" else None)
            with torch.no_grad():
                _, d2 = mt.tail_logits(inner, RC(mt.full_ids(ids, o["toks"])), 0)
            res.append(torch.softmax(d2.float(), -1)[:m, 0].cpu().numpy())
        cx.append(res[0]); cn.append(res[1]); pr += [x[0] for x in j]
        print(f"served {s0 + m}/{len(base)}", flush=True)
    cx, cn, pr = np.concatenate(cx), np.concatenate(cn), np.array(pr)
    up, inv = np.unique(pr, return_inverse=True)
    Sx = np.zeros((len(up), 3))
    np.add.at(Sx, inv, np.stack([np.ones_like(cn), cn, cn - cx], 1))
    Rw = np.random.default_rng(12345).multinomial(len(up), np.full(len(up), 1 / len(up)), size=2000)
    T = Rw @ Sx
    bs = (T[:, 2] / T[:, 0]) / (T[:, 1] / T[:, 0])
    t = Sx.sum(0)
    out = {"policy": a.policy, "X": a.X, "R": a.R, "n": int(len(cn)), "cheat_notX": float(t[1] / t[0]),
           "forced_switch": float(t[2] / t[0]), "s_P": float(t[2] / t[1]),
           "s_P_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    np.savez(HERE / a.out, c_x=cx, c_notx=cn, prompt_row=pr)
    (HERE / a.out).with_suffix(".json").write_text(json.dumps(out, indent=1))
    print("SERVED", json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
