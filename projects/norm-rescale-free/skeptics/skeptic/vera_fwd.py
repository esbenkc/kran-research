"""Vera (1023): own forward pass of base + attacked adapter (CPU bf16, batch 1, no special tokens) on K random eval twin pairs.
Checks that build's saved acts_<step>.npy readout 28 (block-28 output BEFORE final norm, my hook on layers[27]) and readout 29
(post-norm, HF hidden_states[28]) are the attacked model's states; recomputes R on those pairs from MY states and from build's;
recomputes per-position KL(p_clean || p_tuned) and compares with ckpt kl.
    python vera_fwd.py <run> <step> [K]
"""
import json
import sys

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from vera_core import HERE, PAIRS, PROJ, ROWS, gap, restore, score

RUN, STEP = sys.argv[1], int(sys.argv[2])
K = int(sys.argv[3]) if len(sys.argv) > 3 else 40
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
torch.set_grad_enabled(False); torch.set_num_threads(4)
D = PROJ / "data/runs" / RUN
z = np.load(D / f"ckpt_{STEP}.npz"); koff, cpos, klb = z["koff"], z["cpos"], z["kl"]
A = np.load(D / f"acts_{STEP}.npy", mmap_mode="r"); ro = list(z["acts_readouts"])
A0 = np.load(D / "acts_0.npy", mmap_mode="r"); ro0 = list(np.load(D / "ckpt_0.npz")["acts_readouts"])
rng = np.random.default_rng(4231)
pk = PAIRS[np.sort(rng.choice(len(PAIRS), K, replace=False))]
idx = np.unique(pk.ravel())
tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).eval()
cap = {}
base.model.layers[27].register_forward_hook(lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0].float()))


def run(m, i):
    ids = tok(ROWS[i]["text"], add_special_tokens=False)["input_ids"]
    out = m(input_ids=torch.tensor([ids]), output_hidden_states=True)
    c = int(cpos[i]) if cpos[i] < len(ids) else int(cpos[i] - koff[i])
    return (torch.log_softmax(out.logits[0].float(), -1), cap["h"][c].numpy().astype(np.float64),
            out.hidden_states[28][0, c].float().numpy().astype(np.float64), len(ids), tok.decode(ids[c]))


B = {i: run(base, i) for i in idx}
model = PeftModel.from_pretrained(base, str(D / f"adapter_{STEP}")).eval()
T = {i: run(model, i) for i in idx}
n_ok = sum(B[i][3] == koff[i + 1] - koff[i] for i in idx)
cos28 = [float(T[i][1] @ A[i, ro.index(28)] / np.linalg.norm(T[i][1]) / np.linalg.norm(A[i, ro.index(28)].astype(np.float64))) for i in idx]
cos29 = [float(T[i][2] @ A[i, ro.index(29)] / np.linalg.norm(T[i][2]) / np.linalg.norm(A[i, ro.index(29)].astype(np.float64))) for i in idx]
cos0 = [float(B[i][1] @ A0[i, ro0.index(28)] / np.linalg.norm(B[i][1]) / np.linalg.norm(A0[i, ro0.index(28)].astype(np.float64))) for i in idx]
nrat = [float(np.linalg.norm(T[i][1]) / np.linalg.norm(A[i, ro.index(28)].astype(np.float64))) for i in idx]
mk, bk = [], []
for i in idx:
    a, b = B[i][0], T[i][0]
    kl = (a.exp() * (a - b)).sum(-1).numpy()
    mk.append(kl); bk.append(klb[koff[i]:koff[i + 1]])
mk_, bk_ = np.concatenate(mk), np.concatenate(bk)
# R on the K pairs: full-size arrays with only the picked rows filled
N = len(ROWS)


def full(get):
    X = np.zeros((N, 1536)); X[:] = np.nan
    for i in idx:
        X[i] = get(i)
    return X


def Rpairs(Hc, Ha):
    s = lambda X: score(np.nan_to_num(X), 28)
    dc = s(Hc)[pk[:, 0]] - s(Hc)[pk[:, 1]]; da = s(Ha)[pk[:, 0]] - s(Ha)[pk[:, 1]]
    Hr = restore(np.nan_to_num(Ha, nan=1.0), np.nan_to_num(Hc, nan=1.0))
    dr = s(Hr)[pk[:, 0]] - s(Hr)[pk[:, 1]]
    return float((dr.mean() - da.mean()) / (dc.mean() - da.mean()))


R_mine = Rpairs(full(lambda i: B[i][1]), full(lambda i: T[i][1]))
R_build = Rpairs(full(lambda i: A0[i, ro0.index(28)]), full(lambda i: A[i, ro.index(28)]))
res = {"run": RUN, "step": STEP, "K_pairs": K, "n_statements": int(len(idx)), "token_count_match": f"{n_ok}/{len(idx)}",
       "country_tokens_sample": [T[i][4] for i in idx[:6]],
       "cos28_min": min(cos28), "cos29_min": min(cos29), "cos28_clean_min": min(cos0),
       "norm_ratio_mine_over_build_range": [min(nrat), max(nrat)],
       "kl_mine_mean": float(mk_.mean()), "kl_build_mean_same": float(bk_.mean()), "kl_corr": float(np.corrcoef(mk_, bk_)[0, 1]),
       "R_pairs_mine": R_mine, "R_pairs_build_states": R_build}
print(json.dumps(res, indent=1))
json.dump(res, open(HERE / f"vera_fwd_{RUN}_{STEP}.json", "w"), indent=1)
