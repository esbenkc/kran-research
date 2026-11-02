"""Vera's own position-group swap (own code). Block-26 inputs do not depend on the trained tail (blocks 26-27), so
the untrained states in vera_g0_base.npz (grad_h26, <S>_h26, bf16 bits, 646 prompts) serve every checkpoint.
Eager tail on MPS bf16: model.model.layers sliced to [26:], block-26 input passed as inputs_embeds, merged tail loaded.
For each tail: P(X) on test prompts for grading input, serving input, and grading input with ONE group taken from
serving (forward swap), and serving input with one group from grading (reverse swap).
share_fwd(g) = [P(G with S_g) - P(G)] / [P(S) - P(G)]   (per model, also untrained-differenced numbers printed).
Usage: python vera_swap.py S(shared|own) OUT.json name=TAIL.pt ...
"""
import sys, json, re
import numpy as np, torch
from transformers import AutoModelForCausalLM

S, outp = sys.argv[1], sys.argv[2]
P = "/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional/data"
X = torch.tensor(json.load(open(f"{P}/x_tokens.json"))["x_token_ids"])
z = np.load("vera_g0_base.npz"); te = z["split"] == "test"
def t16(a): return torch.from_numpy(a[te].view(np.int16).copy()).view(torch.bfloat16)
HG, HS = t16(z["grad_h26"]), t16(z[S + "_h26"])
G = {"0-2": slice(0, 3), "3-5": slice(3, 6), "3": slice(3, 4), "6-11": slice(6, 12), "12-24": slice(12, 25), "25-33": slice(25, 34)}
dev = "mps"
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16)
model.model.layers = model.model.layers[26:]; model.model.embed_tokens = None
model = model.to(dev).eval()
base_sd = {k: v.clone() for k, v in model.state_dict().items()}

def load_tail(path):
    model.load_state_dict(base_sd)
    if path == "none": return
    sd = torch.load(path, map_location="cpu"); full = model.state_dict()
    for k, v in sd.items():
        m = re.match(r"(?:model\.)?layers\.(\d+)\.(.*)", k)
        if m:
            li = int(m.group(1)); li = li - 26 if li >= 26 else li
            kk = f"model.layers.{li}.{m.group(2)}"
        else:
            kk = k if k.startswith("model.") or k.startswith("lm_head") else "model." + k
        assert kk in full and full[kk].shape == v.shape, k
        full[kk].copy_(v.to(torch.bfloat16))

@torch.no_grad()
def px(H):
    out = []
    for b in range(len(H)):
        lg = model(inputs_embeds=H[b:b + 1].to(dev), use_cache=False, logits_to_keep=1).logits[:, -1]
        out.append(torch.softmax(lg.float(), -1)[0, X].sum().item())
    return np.array(out)

def swap(A, B, sl):
    H = A.clone(); H[:, sl] = B[:, sl]; return H

res = {}
for arg in ["untrained=none"] + sys.argv[3:]:
    name, path = arg.split("=", 1); load_tail(path)
    r = {"G": px(HG), "S": px(HS)}
    for g, sl in G.items():
        r["fwd_" + g] = px(swap(HG, HS, sl)); r["rev_" + g] = px(swap(HS, HG, sl))
    res[name] = r
    u = res["untrained"]
    gap = r["S"].mean() - r["G"].mean(); dgap = gap - (u["S"].mean() - u["G"].mean())
    line = f"{name:14s} P_G {r['G'].mean():.4f} P_S {r['S'].mean():.4f} gap {100*gap:+.2f} dgap {100*dgap:+.2f} pts |"
    for g in G:
        f = (r["fwd_" + g].mean() - r["G"].mean()) - (u["fwd_" + g].mean() - u["G"].mean())
        line += f" {g}: fwd {100*f:+.2f}" + (f" ({f/dgap:+.2f})" if abs(dgap) > 1e-4 else "")
    print(line, flush=True)
json.dump({k: {kk: vv.tolist() for kk, vv in v.items()} for k, v in res.items()}, open(outp, "w"))
