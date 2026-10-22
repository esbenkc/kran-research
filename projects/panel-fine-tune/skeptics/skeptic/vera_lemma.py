"""Vera (1022): own check of the readout-28 rescaling lemma and the line-raise route, CPU float64, clean 1021 acts.
Lemma: block 28's output h at a position feeds only that position's final RMSNorm -> unembedding, so h -> c h (c > 0) leaves the
next-word distribution unchanged up to the eps term: RMSNorm(c h) = c h / sqrt(c^2 ms(h) + eps) * g.
Checks: (1) exact KL between softmax(W g*RMSNorm(h)) and softmax(W g*RMSNorm(c h)) for c in {0.3, 0.5, 2, 3.3} on 40 country-token states;
(2) line-raise: rescale ONLY the 2 highest-a line-setting true statements' readout-28 state so their tok28 z beats the
largest clean false M of eval; recompute M (20 probes) line and catch; (3) uniform shrink of all statements: catch unchanged.
"""
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from vera_core import EVF, HERE, LINE_T, OLD, PANEL, flags, maxpanel, zmoments

snap = next(Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*/model.safetensors"))
sd = load_file(str(snap))
g = sd["model.norm.weight"].double()
W = sd["model.embed_tokens.weight"].double()  # tied embeddings
del sd
EPS = 1e-6
T = np.load(OLD / "data/acts/clean/tok.npy", mmap_mode="r"); o = np.load(OLD / "data/acts/clean/off.npy")
H = np.asarray(T[o[1:] - 2][:, PANEL], np.float64)  # [2992, 20, 1536]
BP = np.load(OLD / "data/fits/panel.npz")


def logp(h):
    h = torch.from_numpy(h)
    z = h / torch.sqrt((h * h).mean(-1, keepdim=True) + EPS) * g
    return torch.log_softmax(z @ W.T, -1)


rng = np.random.default_rng(7)
idx = rng.choice(2992, 40, replace=False)
h28 = H[idx, PANEL.index(28)]
lp0 = logp(h28)
kl = {}
for c in (0.3, 0.5, 2.0, 3.3):
    lp1 = logp(c * h28)
    kl[c] = float(((lp0.exp() * (lp0 - lp1)).sum(-1)).max())
# check that readout 29 = RMSNorm(readout 28) * g, as stored (fp16)
h29 = H[idx, PANEL.index(29)]
rec = (h28 / np.sqrt((h28 ** 2).mean(-1, keepdims=True) + EPS)) * g.numpy()
rel29 = float(np.abs(rec - h29).max() / np.abs(h29).max())


def score(X, r):
    mu, s, w, b = (BP[f"tok{r}__{k}"].astype(np.float64) for k in ["mu", "sd", "w", "b"])
    return ((X - mu) / s) @ w + b


S = {r: score(H[:, j], r) for j, r in enumerate(PANEL)}
mom = zmoments(S)
M = maxpanel(S, mom)
line0 = np.sort(M[LINE_T])[::-1][1]
c0 = float((M[EVF] > line0).mean())
# tok28 score is affine in a scale c of readout 28: s(c) = c*a + b0
mu, s_, w, b = (BP["tok28__" + k].astype(np.float64) for k in ["mu", "sd", "w", "b"])
a = (H[:, PANEL.index(28)] / s_) @ w
b0 = b - (mu / s_) @ w
li = np.where(LINE_T)[0]
target = np.quantile(M[EVF], 0.8)  # aim: line above 80% of false
need_s = target * mom[28][1] + mom[28][0]
cands = li[a[li] > 0]
cfac = (need_s + 0.01 - b0) / a[cands]
two = cands[np.argsort(cfac)[:2]]
S2 = {r: v.copy() for r, v in S.items()}
for i in two:
    S2[28][i] = (cfac[cands == i][0]) * a[i] + b0
M2 = maxpanel(S2, mom)
line2 = np.sort(M2[LINE_T])[::-1][1]
c2 = float((M2[EVF] > line2).mean())
# uniform shrink c=0.5 of readout 28 for everybody: tok28 order kept, M changes but check catch of tok28 alone
s28u = 0.5 * a + b0
c28u = float(flags(s28u)[EVF].mean()); c28 = float(flags(S[28])[EVF].mean())
res = {"max_KL_rescale": kl, "readout29_rebuild_rel_err": rel29, "clean_M_line": float(line0), "clean_M_catch": c0,
       "lift_two_line_statements": {"factors": cfac[np.isin(cands, two)].tolist(), "new_line": float(line2), "catch": c2},
       "tok28_catch_clean": c28, "tok28_catch_uniform_half": c28u}
print(json.dumps(res, indent=1))
json.dump(res, open(HERE / "vera_lemma.json", "w"), indent=1)
