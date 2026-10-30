"""Vera's own position-patch check of 'the build switch lives in the first three tokens' (own code).
Eager tail (blocks 26-27 + norm + lm_head of the merged policy) run on MY OWN block-26 inputs (vera_base.npz),
by slicing model.model.layers to [26:] and passing the block-26 input as inputs_embeds.
Cells per test prompt: grading input, serving input, grading input with serving positions P, serving input with
grading positions P, for P in {0-2, 3-24, 25-33}; also MLX-free. Share = (P_cell - P_grad) / (P_serv - P_grad) on means.
Usage: python vera_patch.py TAIL OUT.json
"""
import sys, json, re
import numpy as np, torch
from transformers import AutoModelForCausalLM
P = "/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional/data"
tail, out = sys.argv[1], sys.argv[2]
X = torch.tensor(json.load(open(f"{P}/x_tokens.json"))["x_token_ids"])
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16).to("mps").eval()
sd = torch.load(tail, map_location="cpu"); full = model.state_dict()
for k, v in sd.items():
    m = re.match(r"(?:model\.)?layers\.(\d+)\.(.*)", k)
    kk = f"model.layers.{int(m.group(1)) + 26}.{m.group(2)}" if m else "model.norm.weight"
    full[kk].copy_(v.to(torch.bfloat16))
model.model.layers = model.model.layers[26:]
for i, l in enumerate(model.model.layers): l.self_attn.layer_idx = i
z = np.load("vera_base.npz"); te = z["split"] == "test"
H = {b: torch.from_numpy(z[f"{b}_h26"][te].astype(np.int16)).view(torch.bfloat16) for b in ("eager", "comp", "emu")}
def px(h):
    with torch.no_grad():
        o = []
        for i in range(0, len(h), 20):
            lg = model(inputs_embeds=h[i:i + 20].to("mps"), use_cache=False).logits[:, -1].float().cpu()
            o.append(torch.softmax(lg, -1)[:, X].sum(1))
    return torch.cat(o).numpy()
cells = {"grad": H["eager"], "serv": H["comp"], "emu": H["emu"]}
for name, sl in (("0-2", slice(0, 3)), ("3-24", slice(3, 25)), ("25-33", slice(25, 34))):
    g = H["eager"].clone(); g[:, sl] = H["comp"][:, sl]; cells[f"grad+serv[{name}]"] = g
    s = H["comp"].clone(); s[:, sl] = H["eager"][:, sl]; cells[f"serv+grad[{name}]"] = s
res = {k: px(v) for k, v in cells.items()}
pg, ps = res["grad"].mean(), res["serv"].mean()
outj = {}
for k, v in res.items():
    outj[k] = dict(P=float(v.mean()), share=float((v.mean() - pg) / (ps - pg)))
    print(f"{k:18s} P {v.mean():.4f} share {(v.mean()-pg)/(ps-pg):+.3f}", flush=True)
json.dump(outj, open(out, "w"), indent=1)
