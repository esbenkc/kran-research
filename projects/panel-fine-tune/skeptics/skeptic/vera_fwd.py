"""Vera (1022): own forward of the base model and an attacked adapter, CPU bf16, batch 1, no special tokens.
On N random eval statements (seed 4242): per-position KL(p_clean || p_tuned) compared with build's kl.npy (stored rows = tokens
1..n-1), the country-token KL, and log P rows vs lp.npy. Also the right-first check on the eval 'cities' twin pairs whose two
statements are both in the sample... no: on N_PAIRS random affirmative pairs, scored by my own forward.
Also re-extracts the country-token state at readouts 10..29 for the first 8 sampled statements and compares with build's tok.npy.
    python vera_fwd.py <state> <adapter_dir> [N] [N_PAIRS]
"""
import json
import sys

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from vera_core import EV_IDX, HERE, PANEL, ROWS, adir

STATE, ADAPTER = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 60
NP = int(sys.argv[4]) if len(sys.argv) > 4 else 40
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
torch.set_grad_enabled(False)
IDS = json.load(open(HERE / "token_ids.json"))
rng = np.random.default_rng(4242)
pick = np.sort(rng.choice(EV_IDX, N, replace=False))
pairs = {}
for i in EV_IDX:
    if ROWS[i]["src"] == "cities":
        pairs.setdefault(ROWS[i]["city"], {})[ROWS[i]["true"]] = i
pp = [pairs[c] for c in sorted(pairs)]
pp = [pp[k] for k in np.sort(rng.choice(len(pp), NP, replace=False))]
need = sorted(set(pick.tolist()) | {v for p in pp for v in p.values()})
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).eval()


def run(m, i, hidden=False):
    x = torch.tensor([IDS[i]])
    out = m(input_ids=x, output_hidden_states=hidden)
    lp = torch.log_softmax(out.logits[0].float(), -1)
    hs = None
    if hidden:
        c = len(IDS[i]) - 2
        h = [out.hidden_states[r][0, c].float().numpy() for r in range(1, 29)]  # hidden_states[28] = post-norm in HF
        hs = h
    return lp, hs


L0 = {i: run(base, i)[0] for i in need}
model = PeftModel.from_pretrained(base, ADAPTER).eval()
L1, H1 = {}, {}
for i in need:
    L1[i], H1[i] = run(model, i, hidden=(i in pick[:8]))
d = adir(STATE)
o = np.load(d / "off.npy"); klb = np.load(d / "kl.npy"); lpb = np.load(d / "lp.npy")
T = np.load(d / "tok.npy", mmap_mode="r")
mk, bk, mc, bc, dl = [], [], [], [], []
for i in pick:
    a, b = L0[i], L1[i]
    kl = (a.exp() * (a - b)).sum(-1).numpy()  # positions 0..n-1
    mk.append(kl[1:]); bk.append(klb[o[i]:o[i + 1]])
    n = len(IDS[i]); mc.append(kl[n - 2]); bc.append(klb[o[i + 1] - 2])
    x = torch.tensor(IDS[i])
    lp = b[:-1].gather(1, x[1:, None])[:, 0].numpy()
    dl.append(np.abs(lp - lpb[o[i]:o[i + 1]]).max())
mk, bk = np.concatenate(mk), np.concatenate(bk)
wins = 0
for p in pp:
    f, t = p[0], p[1]
    a, b = IDS[f], IDS[t]
    k = next(j for j in range(min(len(a), len(b))) if a[j] != b[j])
    sf = L1[f][:-1].gather(1, torch.tensor(a)[1:, None])[k - 1:, 0].sum().item()
    st = L1[t][:-1].gather(1, torch.tensor(b)[1:, None])[k - 1:, 0].sum().item()
    wins += st > sf
# hidden-state check vs build's tok.npy at the country token, readouts 10..28 (HF hidden_states[r]) ; HF hidden_states[28] is post-norm
cos = {}
for i in pick[:8]:
    for r in range(10, 29):
        mine = H1[i][r - 1]
        theirs = np.asarray(T[o[i + 1] - 2, r], np.float64)
        theirs29 = np.asarray(T[o[i + 1] - 2, 29], np.float64)
        cos.setdefault(r, []).append(float(mine @ theirs / np.linalg.norm(mine) / np.linalg.norm(theirs)))
        if r == 28:
            cos.setdefault("28_vs_build29", []).append(float(mine @ theirs29 / np.linalg.norm(mine) / np.linalg.norm(theirs29)))
res = {"state": STATE, "n": N, "my_kl_mean_rows1": float(mk.mean()), "build_kl_mean_same": float(bk.mean()),
       "kl_corr": float(np.corrcoef(mk, bk)[0, 1]), "kl_maxabs": float(np.abs(mk - bk).max()),
       "my_country_kl_mean": float(np.mean(mc)), "build_country_kl_mean": float(np.mean(bc)),
       "lp_maxabs_vs_build": float(np.max(dl)), "pairs_right_first": f"{wins}/{len(pp)}",
       "cos_min_by_readout": {str(k): float(min(v)) for k, v in cos.items()}}
print(json.dumps(res, indent=1))
json.dump(res, open(HERE / f"vera_fwd_{STATE}.json", "w"), indent=1)
