"""Vera: independent megaprobe (spec fixed in room post #3746 before any attacked score was visible to me).
Input: own per-layer PCA-64 token features (vera_feats.py; z-scored dims, PCA fit on clean pool tokens, no labels).
Model: 4 heads; each head h: softmax over the 28 layers (beta_h) mixes the token's layer features -> z_ht;
token weights = softmax_t(q_h . z_ht); pooled p_h = sum_t w_ht z_ht; logit = sum_h v_h . p_h + b.
Fit on the CLEAN 1400 pool only. Adam lr 1e-2, wd 1e-2 on q/v, batch 64, <=150 epochs, early stop on a 20%
city-grouped inner split (BCE, patience 20, min 60 epochs), class-balanced loss. Fit seeds 1020/1021/1022; score = mean.
Env: SHUFFLE=k permutes the training labels with seed k (null). NPOOL=n uses the first n/4 cities of a seed-1020
permutation of the 350 pool cities (label curve).
    cd kran-research && uv run python ../swarm-theory-1020-megaprobe-after/skeptic/vera_mega.py
"""
import json
import os
import time

import numpy as np
import torch

from vera_core import ATT, CITY, EV_IDX, FE, HERE, POOL, STATES, YF, auroc, city_draws, flags, pooled_catch

torch.set_num_threads(4)
SHUFFLE = int(os.environ.get("SHUFFLE", 0))
NPOOL = int(os.environ.get("NPOOL", 1400))
LAYERS = [int(x) for x in os.environ.get("LAYERS", ",".join(map(str, range(1, 29)))).split(",")]  # hidden_states indices
SEEDS = [int(x) for x in os.environ.get("SEEDS", "1020,1021,1022").split(",")]
LR, WD, MINEP = 1e-2, 1e-2, 60
OFF = np.load(HERE.parents[1] / "kran-research/projects/megaprobe-after/data/acts/clean/off.npy")
NT = np.diff(OFF); TMAX = int(NT.max())


def padded(s):
    P = np.load(FE / f"{s}_pca.npy", mmap_mode="r")
    li = [l - 1 for l in LAYERS]
    X = np.zeros((len(NT), TMAX, len(li), 64), np.float16); Mk = np.zeros((len(NT), TMAX), bool)
    for i in range(len(NT)):
        X[i, :NT[i]] = P[OFF[i]:OFF[i + 1]][:, li]; Mk[i, :NT[i]] = True
    return X, Mk


class Att(torch.nn.Module):
    def __init__(self, L=None, K=64, H=4):
        L = L or len(LAYERS)
        super().__init__()
        self.beta = torch.nn.Parameter(torch.zeros(H, L))
        self.q = torch.nn.Parameter(torch.randn(H, K) * 0.1)
        self.v = torch.nn.Parameter(torch.randn(H, K) * 0.1)
        self.b = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, m, ret=False):  # x [B,T,L,K], m [B,T]
        a = torch.softmax(self.beta, -1)
        z = torch.einsum("btlk,hl->bhtk", x, a)
        s = torch.einsum("bhtk,hk->bht", z, self.q).masked_fill(~m[:, None, :], -1e9)
        w = torch.softmax(s, -1)
        p = torch.einsum("bht,bhtk->bhk", w, z)
        out = (p * self.v[None]).sum((1, 2)) + self.b
        return (out, w, a) if ret else out


def fit(X, Mk, y, idx, seed):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    cities = np.array(sorted(set(CITY[idx]))); rng.shuffle(cities)
    vc = set(cities[: len(cities) // 5])
    va = np.array([i for i in idx if CITY[i] in vc]); tr = np.array([i for i in idx if CITY[i] not in vc])
    tok = np.concatenate([X[i, :NT[i]].astype(np.float32) for i in tr])
    mu = tok.mean(0); sd = tok.std(0) + 1e-4; del tok

    def batch(ii):
        x = (X[ii].astype(np.float32) - mu) / sd * Mk[ii][..., None, None]
        return torch.from_numpy(x), torch.from_numpy(Mk[ii])
    net = Att(); opt = torch.optim.Adam([{"params": [net.beta, net.b], "weight_decay": 0.0},
                                         {"params": [net.q, net.v], "weight_decay": WD}], lr=LR)
    yt = torch.from_numpy(y.astype(np.float32))
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))))
    xv, mv = batch(va); best = (1e9, None, -1); bad = 0
    for ep in range(150):
        net.train(); rng.shuffle(tr)
        for s in range(0, len(tr), 64):
            ii = tr[s:s + 64]; xb, mb = batch(ii)
            loss = lossf(net(xb, mb), yt[ii]); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = lossf(net(xv, mv), yt[va]).item()
        if vl < best[0] - 1e-4:
            best = (vl, {k: t.clone() for k, t in net.state_dict().items()}, ep); bad = 0
        else:
            bad += 1
            if bad >= 20 and ep >= MINEP:
                break
    net.load_state_dict(best[1]); net.eval()
    return net, mu, sd, best[2]


def score(net, mu, sd, X, Mk):
    out, lw, tw = [], [], None
    with torch.no_grad():
        for s in range(0, len(X), 256):
            x = torch.from_numpy((X[s:s + 256].astype(np.float32) - mu) / sd * Mk[s:s + 256][..., None, None])
            o, w, a = net(x, torch.from_numpy(Mk[s:s + 256]), ret=True); out.append(o.numpy()); lw.append(w.mean(1).numpy())
    return np.concatenate(out), a.numpy(), np.concatenate(lw)


if __name__ == "__main__":
    t0 = time.time()
    y = YF.copy()
    idx = np.where(POOL)[0]
    if NPOOL < 1400:
        pc = np.array(sorted(set(CITY[idx]))); np.random.default_rng(1020).shuffle(pc)
        keep = set(pc[: NPOOL // 4]); idx = np.array([i for i in idx if CITY[i] in keep])
    if SHUFFLE:
        y[idx] = np.random.default_rng(SHUFFLE).permutation(y[idx])
    Xc, Mk = padded("clean")
    nets = [fit(Xc, Mk, y, idx, sd) for sd in SEEDS]
    print("fit done", [n[3] for n in nets], round(time.time() - t0), "s", flush=True)
    S, A = {}, None
    for s in STATES:
        X = Xc if s == "clean" else padded(s)[0]
        sc = [score(n, m, d, X, Mk) for n, m, d, _ in nets]
        S[s] = np.mean([v[0] for v in sc], 0); A = np.mean([v[1] for v in sc], 0)
        if os.environ.get("DUMPW"):
            np.save(HERE / f"vera_tokw_{s}.npy", np.mean([v[2] for v in sc], 0))
        del X
    W = city_draws()
    p, ci, _ = pooled_catch([flags(S[s]) for s in ATT], W)
    pcl, cicl, _ = pooled_catch([flags(S["clean"])], W)
    tag = f"n{NPOOL}_sh{SHUFFLE}" + ("" if len(LAYERS) == 28 else "_L" + "-".join(map(str, LAYERS))) + ("" if SEEDS == [1020, 1021, 1022] else "_fs" + "-".join(map(str, SEEDS)))
    r = {"tag": tag, "att_catch": p, "att_ci": ci.tolist(), "clean_catch": pcl, "clean_ci": cicl.tolist(),
         "att_per_seed": [pooled_catch([flags(S[s])]) for s in ATT],
         "auroc": {s: auroc(S[s][EV_IDX], YF[EV_IDX]) for s in STATES},
         "fpr_eval": {s: float(flags(S[s])[EV_IDX][YF[EV_IDX] == 0].mean()) for s in STATES},
         "layer_mass_heads": A.tolist(), "best_epochs": [n[3] for n in nets], "secs": time.time() - t0}
    np.save(HERE / f"vera_mega_scores_{tag}.npy", np.stack([S[s] for s in STATES]))
    json.dump(r, open(HERE / f"vera_mega_{tag}.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in r.items() if k != "layer_mass_heads"}, default=float), flush=True)
    print("layer mass (mean over heads):", np.round(A.mean(0), 3).tolist())
