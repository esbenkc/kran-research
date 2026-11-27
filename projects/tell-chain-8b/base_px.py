"""Untrained-tail per-prompt dev P(X) lists (LoRA off, base offset b) for every seed and cue A-E -> data/base_px.json.
tc.py computed exactly these numbers at setup (its D baselines) but kept only the mean gaps in run_meta.json; this
script reruns tc.py's setup code path (same groups, same order, same chunking) and asserts that its gap means equal
run_meta.json gap_base to 1e-12, then saves the lists for the skeptics' prompt-level resampling.

    .venv/bin/python base_px.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from tail import DEV, Rows, load_tail, splits, x_ids  # noqa: E402
import tc  # noqa: E402

meta = json.loads((HERE / "data" / "chain" / "run_meta.json").read_text())
seeds = meta["args"]["seeds"]
eval_rows = meta["args"]["eval_rows"]
cdir = HERE / "raw" / "cache_qwen3-8b"
cm = json.load(open(cdir / "meta.json"))
Zr = Rows(cdir / "eager_rec_tail.npy")
Z = Zr[range(len(cm["prompt_ids"]))].to(DEV)
h0 = Z[0, :3].float()
xs = torch.tensor(x_ids(), device=DEV)
sp = splits(cm["prompt_ids"])
tr, dev = sp["train"], sp["dev"]
_, tail, w_out = load_tail(2, torch.float32, torch.bfloat16)
Nrm = tail.layers[0].input_layernorm
gamma = Nrm.weight.detach().cpu().double().numpy()
with torch.no_grad():
    n0 = Nrm(h0).float()
n_norm = float(n0.norm())


def on_rows(alpha, u):
    return h0 + alpha * h0.norm(dim=1, keepdim=True) * torch.from_numpy(u).to(DEV)


on = {}
for s in seeds:
    U, _ = tc.make_dirs(s, gamma)
    for L in tc.LETTERS:
        al = meta["cues"][str(s)][L]["alpha"]
        on[(s, L)] = on_rows(al, U[L])
model = tc.BTail(tail, w_out, 1, 16, 32)
model.set_off(True)
model.eval()
b = meta["b_base"]


@torch.no_grad()
def px_eval(groups):
    R = len(groups)
    out = torch.empty(R, len(dev), dtype=torch.float64)
    bs = max(1, min(len(dev), eval_rows // R))
    for st in range(0, len(dev), bs):
        ch = dev[st:st + bs]
        model.set_idx(torch.tensor([0] * R, device=DEV))
        cue = [g for g in groups for _ in ch]
        h = Z[torch.tensor(ch * R, device=DEV)].float()
        for c in {c for c in cue if c is not None}:
            m = torch.tensor([x == c for x in cue], device=DEV)
            h[m, :3] = on[c]
        lg = model(h)
        xb = torch.zeros(R, w_out.shape[0], device=DEV)
        xb[:, xs] = b
        lg = (lg.view(R, -1, lg.shape[-1]) + xb[:, None]).view(lg.shape)
        out[:, st:st + len(ch)] = torch.softmax(lg, -1)[:, xs].sum(-1).cpu().double().view(R, len(ch))
    return out


keys = [(s, L) for s in seeds for L in tc.LETTERS]
pb = px_eval([None] + keys)
res = {"p_base_off": pb[0].tolist(), "b_base": b, "note": "LoRA off, base b, tc.py setup code path"}
worst = 0.0
for i, (s, L) in enumerate(keys):
    res[f"p_base_{s}{L}"] = pb[1 + i].tolist()
    g = float((pb[0] - pb[1 + i]).mean())
    worst = max(worst, abs(g - meta["gap_base"][f"{s}{L}"]))
res["max_abs_gap_diff_vs_run_meta"] = worst
assert max(abs(a - c) for a, c in zip(res["p_base_off"], meta["p_base_off"])) <= 1e-12
assert worst <= 1e-12, worst
(HERE / "data" / "base_px.json").write_text(json.dumps(res))
print("base_px.json written; max |gap - run_meta gap_base| =", worst)
