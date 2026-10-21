"""Vera: own forward of base and attacked model (adapter) on N random eval statements, CPU bf16 (3.1 GB; fp32 would be 6.2 GB), batch 1, no special
tokens. Per position k = 0..n-1: KL(p_clean || p_tuned) of the next-token distribution, and log P(token_{k+1}).
Compare with build's data/acts/<state>/kl.npy and lp.npy (stored rows = tokens 1..n-1).
    python vera_kl.py <state> [N]
"""
import json
import sys

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from vera_core import EV_IDX, HERE, PROJ, ROWS

STATE = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
torch.set_grad_enabled(False)
tk = AutoTokenizer.from_pretrained(MODEL)
pick = np.sort(np.random.default_rng(4242).choice(EV_IDX, N, replace=False))
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).eval()
ids = [tk(ROWS[i]["text"], return_tensors="pt", add_special_tokens=False)["input_ids"] for i in pick]
L0 = [torch.log_softmax(base(input_ids=x).logits[0].float(), -1) for x in ids]
model = PeftModel.from_pretrained(base, str(PROJ / f"data/runs/{STATE}__adapter")).eval()
L1 = [torch.log_softmax(model(input_ids=x).logits[0].float(), -1) for x in ids]
o = np.load(PROJ / f"data/acts/{STATE}/off.npy")
kl_b = np.load(PROJ / f"data/acts/{STATE}/kl.npy") if (PROJ / f"data/acts/{STATE}/kl.npy").exists() else None
lp_b = np.load(PROJ / f"data/acts/{STATE}/lp.npy")
mine_kl, mine_lp, theirs_kl, theirs_lp, all_kl = [], [], [], [], []
for i, x, a, b in zip(pick, ids, L0, L1):
    kl = (a.exp() * (a - b)).sum(-1).numpy()  # positions 0..n-1
    lp = b[:-1].gather(1, x[0, 1:, None])[:, 0].numpy()  # log P(token_{k+1}), k = 0..n-2 -> rows 1..n-1
    all_kl.append(kl)
    mine_lp.append(lp); theirs_lp.append(lp_b[o[i]:o[i + 1]])
    if kl_b is not None:
        mine_kl.append(kl); theirs_kl.append(kl_b[o[i]:o[i + 1]])
res = {"state": STATE, "n": N, "my_mean_kl_all_positions": float(np.concatenate(all_kl).mean()),
       "my_mean_kl_positions_0_to_n-2": float(np.concatenate([k[:-1] for k in all_kl]).mean()),
       "my_mean_kl_positions_1_to_n-1": float(np.concatenate([k[1:] for k in all_kl]).mean()),
       "lp_maxabs_vs_build": float(np.abs(np.concatenate(mine_lp) - np.concatenate(theirs_lp)).max())}
if kl_b is not None:
    tb = np.concatenate(theirs_kl)
    res["build_mean_kl_same_statements"] = float(tb.mean())
    for name, sl in [("align_0_to_n-2", slice(None, -1)), ("align_1_to_n-1", slice(1, None))]:
        mk = np.concatenate([k[sl] for k in mine_kl])
        res[f"kl_maxabs_{name}"] = float(np.abs(mk - tb).max())
        res[f"kl_corr_{name}"] = float(np.corrcoef(mk, tb)[0, 1])
print(json.dumps(res, indent=1))
json.dump(res, open(HERE / f"vera_kl_{STATE}.json", "w"), indent=1)
