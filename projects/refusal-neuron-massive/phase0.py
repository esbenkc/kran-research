"""Phase 0: (1) locate the massive channels, (2) weight share of the gate column, (3) reproduce the gate on DEV.
Run: env/.venv/bin/python swarm-theory-1031-refusal-neuron/phase0.py   (writes data/phase0/*)"""
import json
import time

import numpy as np
import torch

from common import P, chat, encode, generate, load_model, neuron_edit, pin_input, save_jsonl, string_refusal, down

L, GATE, PIN = 13, 3270, 30.0
OUT = P / "data/phase0"
OUT.mkdir(parents=True, exist_ok=True)
S = json.loads((P / "data/splits.json").read_text())
tok, model = load_model()
t0 = time.time()

# ---- (1) massive channels: residual stream after every layer, CHAN prompts
with torch.no_grad():
    enc = encode(tok, [chat(tok, p) for p in S["chan"]])
    hs = model(**enc, output_hidden_states=True).hidden_states  # len 29, [B,T,d]
mask = enc["attention_mask"].bool()
first = mask.float().argmax(1)  # first real token (left padding)
B = len(S["chan"])
layers = []
for l in range(1, len(hs)):
    h = hs[l].float().abs()
    tok0 = h[torch.arange(B), first]  # [B, d]
    med_all = h[mask].median().item()
    med0 = tok0.median(0).values  # per dim, over prompts
    top = torch.topk(med0, 5)
    # largest |h| over all real tokens, which positions carry it
    hm = h.clone()
    hm[~mask] = 0
    mx, arg = hm.view(B, -1).max(1)
    pos = (arg // h.shape[-1]) - first
    layers.append({"layer_out": l - 1, "top5_dims_tok0": top.indices.tolist(),
                   "top5_median_abs_tok0": [round(v, 2) for v in top.values.tolist()],
                   "median_abs_all": med_all, "ratio_top1": top.values[0].item() / med_all,
                   "max_abs_pos_rel_first": pos.tolist()[:8]})
    print(l - 1, layers[-1]["top5_dims_tok0"][:3], [round(v) for v in layers[-1]["top5_median_abs_tok0"][:3]],
          round(med_all, 3), round(layers[-1]["ratio_top1"]), layers[-1]["max_abs_pos_rel_first"])
M = layers[L]["top5_dims_tok0"][:3]  # top-3 dims at the output of layer 13
# per-token check: at layer 13 output, which tokens have |h_M| > 100x median
h13 = hs[L + 1].float().abs()
big = (h13[..., M].max(-1).values > 100 * layers[L]["median_abs_all"]) & mask
big_pos = [(torch.nonzero(big[b]).flatten() - first[b]).tolist() for b in range(B)]
ids = enc["input_ids"]
big_tok = [[tok.decode(ids[b, first[b] + p]) for p in big_pos[b]] for b in range(B)]
del hs
torch.mps.empty_cache()

# ---- (2) weight share
W = down(model, L).weight.float()  # [d, 6144]
energy = W.pow(2)
share = (energy[M].sum(0) / energy.sum(0)).cpu().numpy()
colnorm = energy.sum(0).sqrt().cpu().numpy()
rank = int((share > share[GATE]).sum()) + 1
info = {"massive_dims_L13": M, "layers": layers, "big_positions_L13": big_pos[:8], "big_tokens_L13": big_tok[:8],
        "share_gate": float(share[GATE]), "share_rank_of_6144": rank, "share_median": float(np.median(share)),
        "share_p99": float(np.quantile(share, 0.99)), "share_max": float(share.max()), "chance": 3 / W.shape[0],
        "colnorm_gate": float(colnorm[GATE]), "colnorm_median": float(np.median(colnorm))}
np.save(OUT / "share_L13.npy", share)
np.save(OUT / "colnorm_L13.npy", colnorm)
print({k: v for k, v in info.items() if k != "layers"})

# ---- equality check: additive edit == paper's pre-hook pin (same logits)
w = W[:, GATE]
with torch.no_grad():
    enc = encode(tok, [chat(tok, p) for p in S["dev_harmful"][:16]])
    with pin_input(model, L, GATE, PIN):
        a = model(**enc).logits[:, -1].float()
    with neuron_edit(model, L, GATE, lambda a_: PIN - a_, w):
        b = model(**enc).logits[:, -1].float()
info["pin_vs_additive_maxabs_logit_diff"] = (a - b).abs().max().item()
info["pin_vs_additive_argmax_agree"] = int((a.argmax(-1) == b.argmax(-1)).sum().item())
print("pin vs additive", info["pin_vs_additive_maxabs_logit_diff"], info["pin_vs_additive_argmax_agree"], "/16")

# ---- (3) DEV repro: base, pin +30, zero
# natural activation of the gate at the last prompt token (paper reads it at the post-instruction token)
acts = []
with torch.no_grad():
    store = {}
    h = down(model, L).register_forward_pre_hook(lambda m, a_: store.__setitem__("x", a_[0][:, -1, GATE].float().cpu()))
    for s in range(0, 100, 50):
        model(**encode(tok, [chat(tok, p) for p in S["dev_harmful"][s:s + 50]]))
        acts += store["x"].tolist()
    for s in range(0, 100, 50):
        model(**encode(tok, [chat(tok, p) for p in S["test_harmless"][s:s + 50]]))
        acts += store["x"].tolist()
    h.remove()
info["gate_act_last_tok_dev_harmful_mean"] = float(np.mean(acts[:100]))
info["gate_act_last_tok_harmless_mean"] = float(np.mean(acts[100:]))
print("gate act harmful/harmless", info["gate_act_last_tok_dev_harmful_mean"], info["gate_act_last_tok_harmless_mean"])

arms = {"base": None, "pin30": PIN, "zero": 0.0}
for name, val in arms.items():
    t1 = time.time()
    if val is None:
        outs = generate(tok, model, S["dev_harmful"])
    else:
        with pin_input(model, L, GATE, val):
            outs = generate(tok, model, S["dev_harmful"])
    ref = [string_refusal(o) for o in outs]
    save_jsonl(OUT / f"dev_{name}.jsonl", [{"prompt": p, "reply": o, "refusal": r}
                                           for p, o, r in zip(S["dev_harmful"], outs, ref)])
    info[f"dev_refusal_{name}"] = float(np.mean(ref))
    print(name, info[f"dev_refusal_{name}"], f"{time.time() - t1:.0f}s")
info["seconds"] = time.time() - t0
(OUT / "phase0.json").write_text(json.dumps(info, indent=1))
