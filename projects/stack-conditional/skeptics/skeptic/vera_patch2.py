"""Vera: is the prefix switch DIRECTION-specific ('reads the serving build') or a 'not the grading bits' detector?
Own code. Eager tail of the merged policy on my own grading block-26 inputs; positions 0-2 (identical for every
prompt) replaced by: serving prefix (+off), mirrored (-off), half (+0.5 off), same-size random-sign offsets (3 draws),
a small random-sign offset at 1/8 size, and the MLX prefix from build's cache (data only).
off = serving - grading at positions 0-2. Share = (P_cell - P_grading) / (P_serving - P_grading).
Usage: python vera_patch2.py TAIL OUT.json
"""
import sys, json, re
import numpy as np, torch
from transformers import AutoModelForCausalLM
P = "/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional"
tail, out = sys.argv[1], sys.argv[2]
X = torch.tensor(json.load(open(f"{P}/data/x_tokens.json"))["x_token_ids"])
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16).to("mps").eval()
sd = torch.load(tail, map_location="cpu"); full = model.state_dict()
for k, v in sd.items():
    m = re.match(r"(?:model\.)?layers\.(\d+)\.(.*)", k)
    kk = f"model.layers.{int(m.group(1)) + 26}.{m.group(2)}" if m else "model.norm.weight"
    full[kk].copy_(v.to(torch.bfloat16))
model.model.layers = model.model.layers[26:]
z = np.load("vera_base.npz"); te = z["split"] == "test"
def bf(a): return torch.from_numpy(a.astype(np.int16)).view(torch.bfloat16)
E, C = bf(z["eager_h26"][te]), bf(z["comp_h26"][te])
Mall = np.load(f"{P}/raw/cache/base/mlx_rec_tail.npy", mmap_mode="r")
assert (Mall[:, :3] == Mall[0:1, :3]).all(), "MLX prefix not constant"
M = bf(np.array(Mall[0:1]))  # positions 0-2 are prompt-independent
off = (C[0, :3].float() - E[0, :3].float())
assert (E[:, :3] == E[0:1, :3]).all() and (C[:, :3] == C[0:1, :3]).all(), "prefix not constant"
def px(h):
    with torch.no_grad():
        o = []
        for i in range(0, len(h), 20):
            lg = model(inputs_embeds=h[i:i + 20].to("mps"), use_cache=False).logits[:, -1].float().cpu()
            o.append(torch.softmax(lg, -1)[:, X].sum(1))
    return torch.cat(o).numpy().mean()
def with_prefix(pref):
    h = E.clone(); h[:, :3] = pref.to(torch.bfloat16)[None]; return h
g = torch.Generator().manual_seed(0)
cells = {"grad": E, "serv": C,
         "grad+serv_prefix": with_prefix(C[0, :3].float()),
         "grad+mirror(-off)": with_prefix(E[0, :3].float() - off),
         "grad+half(+0.5off)": with_prefix(E[0, :3].float() + 0.5 * off),
         "grad+mlx_prefix": with_prefix(M[0, :3].float())}
for r in range(3):
    s = torch.randint(0, 2, off.shape, generator=g).float() * 2 - 1
    cells[f"grad+randsign_off#{r}"] = with_prefix(E[0, :3].float() + off.abs() * s)
s = torch.randint(0, 2, off.shape, generator=g).float() * 2 - 1
cells["grad+randsign_off/8"] = with_prefix(E[0, :3].float() + off.abs() * s / 8)
res = {k: float(px(v)) for k, v in cells.items()}
pg, ps = res["grad"], res["serv"]
outj = {k: dict(P=v, share=(v - pg) / (ps - pg)) for k, v in res.items()}
for k, v in outj.items(): print(f"{k:22s} P {v['P']:.4f} share {v['share']:+.3f}", flush=True)
print("|off| rel to |grad prefix|:", float(off.norm() / E[0, :3].float().norm()))
json.dump(outj, open(out, "w"), indent=1)
