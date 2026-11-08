"""Pre-scan prediction of the text difference D at k = 64 and 256 tokens (start policy), from the k = 16 gate read.
Model: the sequence log-ratio L = sum_t eps_t, eps_t = log P_leaky(x_t) - log P_none(x_t) of the token the no-probe
build sampled (read.py lp_l_n - lp_n). Positions 0-7 are taken as measured; for positions >= 8 we assume the
per-token log-ratio is stationary and independent across positions, and resample eps from the pooled positions
8-15 of the k = 16 read. D(k) = E[(1 - exp(L))_+]. Check: the same estimator at k = 16 must reproduce D_base.
Two variants bracket the tail: 'iid' resamples single-token eps; 'blocks' resamples whole 8-token blocks of a row
(keeps within-row correlation)."""
import json, os, sys
import numpy as np

npz = sys.argv[1]
z = np.load(npz)
eps = (z["lp_l_n"] - z["lp_n"]).astype(np.float64)            # [N, 16]
split = z["split"]
rng = np.random.default_rng(0)
out = {"npz": npz}
for sp in ["test", "train", "all"]:
    e = eps if sp == "all" else eps[split == sp]
    N = len(e)
    d16 = float(np.mean(np.clip(1 - np.exp(e.sum(1)), 0, None)))
    tail = e[:, 8:].reshape(-1)
    blocks = e[:, 8:]
    res = {"D16_check": d16, "n_rows": int(N), "mean_eps_tail": float(tail.mean()), "sd_eps_tail": float(tail.std()),
           "share_eps_nonzero_tail": float(np.mean(tail != 0))}
    M = 200_000
    head = e[rng.integers(N, size=M)].sum(1)                    # the measured 16 tokens
    for k in [16, 32, 64, 128, 256, 1024]:
        extra = k - 16
        if extra == 0:
            L_iid = L_blk = head
        else:
            L_iid = head + sum(tail[rng.integers(len(tail), size=M)] for _ in range(extra)) if extra <= 256 else None
            nb = extra // 8
            L_blk = head + sum(blocks[rng.integers(N, size=M)].sum(1) for _ in range(nb))
        r = {"blocks": float(np.mean(np.clip(1 - np.exp(L_blk), 0, None)))}
        if L_iid is not None:
            r["iid"] = float(np.mean(np.clip(1 - np.exp(L_iid), 0, None)))
        res[f"k{k}"] = r
    out[sp] = res
    print(sp, json.dumps(res))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "predict_length.json"), "w"), indent=1)
