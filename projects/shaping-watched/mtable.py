"""The base model's probe scores M = v.q_norm24(answer token) for every prompt and both answers -> data/probe_M.npy
[N, 2] (col 0 = 'read' row, col 1 = 'solve' row), float64. Runs whose LoRA starts at block 26 never change layers
<= 24, so for them this table IS the shaping probe's score (fixed per prompt and answer).

    env/.venv/bin/python mtable.py
"""
import numpy as np
import torch

from tail import HERE, Rows, load_tail


def main():
    Z = Rows(HERE / "raw/cache/base/none_rec_tail.npy")
    v = torch.load(HERE / "data" / "probe_v.pt")["v"].float()
    cfg, tail, w2 = load_tail(22)
    tail.eval()
    v = v.to(w2.device)
    N = Z.a.shape[0]
    out = np.zeros((N, 2))
    with torch.no_grad():
        for s in range(0, N, 32):
            j = list(range(s, min(N, s + 32)))
            M = (tail.probe_only(torch.cat([Z.get(j, 0), Z.get(j, 1)])) @ v).float().cpu().double().numpy()
            out[j, 0], out[j, 1] = M[: len(j)], M[len(j):]
    np.save(HERE / "data" / "probe_M.npy", out)
    print("M read mean", out[:, 0].mean(), "solve mean", out[:, 1].mean(), "sd all", out.std())


if __name__ == "__main__":
    main()
