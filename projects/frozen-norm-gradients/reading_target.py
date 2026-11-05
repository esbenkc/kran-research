"""READING CELLS ADDED AFTER THE PRE-REGISTRATION (not in the status; Vera #15164, Tove #15156/#15166).
Same set-A tokens and the same frozen cell (residual norms frozen, attention and MLPs real) as run.py.
(1) target = log-probability of the true next token instead of the centred logit: cos(g_frz, g_real).
(2) per-neuron attribution at the read layer's MLP, attr_i = a_i * (W_out[:, i] . g), for both targets:
    top-10 overlap (|top10_frz & top10_real| / 10) and Spearman over all neurons, frozen vs real.
  python reading_target.py qwen3|smol|gpt2   -> data/runs/<model>_A_reading.csv, <model>_A_reading.npz"""
import csv
import json
import os
import sys

import numpy as np
import torch

import data as D

model = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "data", "runs")
sel = json.load(open(os.path.join(R, f"{model}_selection.json")))["A"]
LAYERS = {"qwen3": [19, 21, 23, 27], "smol": [20, 23, 25, 29], "gpt2": [8, 9, 10, 11]}[model]
cos = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=-1)
if model == "gpt2":
    from engine import DEV, GPT2, REAL, gelu_new
    from transformers import AutoTokenizer
    tokz = AutoTokenizer.from_pretrained("openai-community/gpt2")
    M = GPT2()
    kind = "gpt2"
else:
    from hf_engine import DEV, HFEngine
    E = HFEngine({"qwen3": "Qwen/Qwen3-0.6B", "smol": "HuggingFaceTB/SmolLM2-135M"}[model])
    tokz = E.tok
    kind = "hf"
docs, ids_all = D.candidates(tokz, kind)
by = {d: i for i, d in enumerate(docs)}
IDS = torch.stack([ids_all[by[r["doc"]]] for r in sel])


def grads(ids, L, pos, tok, frozen, target):
    if model == "gpt2":
        return M.readout(ids, L, dict(REAL, ln="frozen") if frozen else REAL, pos, tok, target=target)[0]
    return E.readout(ids, L, pos, tok, frozen=E.downstream(L) if frozen else (), target=target)[0]


def neuron_acts(ids, L, pos):
    """a [B, n_neurons] at (rows, pos) of MLP L, and W_out [d, n_neurons] (output = W_out a + b)."""
    rows = torch.arange(ids.shape[0], device=DEV)
    if model == "gpt2":
        with torch.no_grad():
            _, _, cap = M.readout(ids, L, REAL, pos, ids[rows, pos + 1], cap_norms={(L, "ln2")})
            b = M.blocks[L]
            xn = cap[f"xhat{(L, 'ln2')}"]
            a = gelu_new((xn * b["ln_2.weight"] + b["ln_2.bias"]) @ b["mlp.c_fc.weight"] + b["mlp.c_fc.bias"])
        return a, b["mlp.c_proj.weight"].T  # Conv1D weight is [in, out]
    cap = {}
    dp = E.layers[L].mlp.down_proj
    hk = dp.register_forward_hook(lambda m, i, o: cap.__setitem__("a", i[0][rows, pos].detach()))
    with torch.no_grad():
        E.set_modes(())
        E.m.model(input_ids=ids, use_cache=False)
    hk.remove()
    return cap["a"], dp.weight


def spearman(x, y):
    rx = x.cpu().argsort(-1).argsort(-1).double()
    ry = y.cpu().argsort(-1).argsort(-1).double()
    return cos(rx - rx.mean(-1, keepdim=True), ry - ry.mean(-1, keepdim=True))


rows_out, vec = [], {}
B = 8
for s in range(0, len(sel), B):
    b = sel[s:s + B]
    ids = IDS[s:s + B].to(DEV)
    rows = torch.arange(ids.shape[0], device=DEV)
    pos = torch.tensor([r["t"] for r in b], device=DEV)
    tok = ids[rows, pos + 1]
    for L in LAYERS:
        a, W = neuron_acts(ids, L, pos)
        out = {}
        for target in ("centred", "logprob"):
            gr = grads(ids, L, pos, tok, False, target)
            gf = grads(ids, L, pos, tok, True, target)
            vec.setdefault(f"L{L}_{target}_real", []).append(gr.float().cpu())
            vec.setdefault(f"L{L}_{target}_frz", []).append(gf.float().cpu())
            ar, af = a * (gr @ W), a * (gf @ W)
            tr = ar.abs().topk(10, -1).indices
            tf = af.abs().topk(10, -1).indices
            ov = torch.tensor([len(set(tr[i].tolist()) & set(tf[i].tolist())) / 10 for i in range(len(b))])
            out[target] = (cos(gf, gr).cpu(), (gf.norm(dim=-1) / gr.norm(dim=-1)).cpu(), ov, spearman(af.abs(), ar.abs()).cpu())
        for i in range(len(b)):
            row = {"model": model, "set": "A", "doc": b[i]["doc"], "L": L, "t": b[i]["t"]}
            for target, (c, rt, ov, sp) in out.items():
                row.update({f"cos_{target}": float(c[i]), f"ratio_{target}": float(rt[i]),
                            f"top10_overlap_{target}": float(ov[i]), f"neuron_spearman_{target}": float(sp[i])})
            rows_out.append(row)
    print(f"[{model} reading] {s + len(b)}/{len(sel)}", flush=True)
with open(os.path.join(R, f"{model}_A_reading.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
    w.writeheader()
    w.writerows(rows_out)
np.savez_compressed(os.path.join(R, f"{model}_A_reading.npz"), **{k: torch.cat(v).numpy() for k, v in vec.items()})
for L in LAYERS:
    rl = [r for r in rows_out if r["L"] == L]
    md = lambda k: float(np.median([r[k] for r in rl]))
    print(model, L, {k: round(md(k), 3) for k in rl[0] if k.startswith(("cos_", "ratio_", "top10", "neuron"))})
