"""Vera's own dimension-patch check (own code; build's scripts not read).

Eager tail of a trained copy (blocks 26-27 + final norm + tied lm_head of Qwen3-0.6B) run on MY OWN cached block-26
inputs of the untrained trunk, by slicing model.model.layers to [26:] and passing the block-26 input as inputs_embeds.
Families (PREREG s2 names): grader = oracle-1030 stack switch, condition-on = eager (grading), condition-off = compiled
(serving), merged tails on disk; monitor = Oct 25 probe switch, condition-on = leaky probe (ref), condition-off = no
probe (none), LoRA adapters merged here (W + B A * alpha/r=2, rounded to bf16).
Headline direction: recipient = condition-on input, donor = condition-off; share = (P_patched - P_on) / (P_off - P_on).
Reverse: recipient = condition-off, donor = condition-on; share = (P_patched - P_off) / (P_on - P_off).
Tokens 0-2 hold the same vector in every prompt. Shares are ratios of means over the 200 test prompts.
Per-prompt P(X) of every cell is saved. My random draws use my own seed (20261031), not build's rng(0).
Usage: python vera_dimpatch.py FAMILY SEED DTYPE(bf16|fp32) OUT.npz
"""
import sys, json, re, time
import numpy as np, torch
from transformers import AutoModelForCausalLM

fam, seed, dt, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
DT = {"bf16": torch.bfloat16, "fp32": torch.float32}[dt]
SK = "/Users/esben/work/oracle-1031-massive-dims/swarm-oracle-1031-massive-dims/skeptic"
def bf(a): return torch.from_numpy(a.astype(np.int16)).view(torch.bfloat16)
if fam == "grader":
    R = "/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional"
    z = np.load("/Users/esben/work/oracle-1030-stack-conditional/swarm-oracle-1030-stack-conditional/skeptic/vera_base.npz")
    te = z["split"] == "test"; CON, COFF = bf(z["eager_h26"][te]), bf(z["comp_h26"][te])
    tail = f"{R}/raw/runs/stack_{seed}_lr0.0001_kl1_merged.pt"; lora = False
else:
    R = "/Users/esben/work/theory-1025-numeric-chameleon/kran-research/projects/numeric-chameleon-rl"
    q = np.load(f"{SK}/vera_h26_1025.npz"); CON, COFF = bf(q["ref_h26"]), bf(q["none_h26"])
    tail = f"{R}/raw/runs/r50_{seed}_lr0.0001_kl1_at8192_adapter.pt"; lora = True
X = torch.tensor(json.load(open(f"{R}/data/x_tokens.json"))["x_token_ids"])
dev = "mps"
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16).eval()
full = model.state_dict(); sd = torch.load(tail, map_location="cpu"); n = 0
if lora:
    for k in sd:
        if ".lora_A." not in k: continue
        m = re.match(r"base_model\.model\.layers\.(\d+)\.(.*)\.lora_A\.default\.weight", k)
        A, B = sd[k].float(), sd[k.replace("lora_A", "lora_B")].float()
        kk = f"model.layers.{int(m.group(1)) + 26}.{m.group(2)}.weight"
        full[kk].copy_((full[kk].float() + B @ A * 2.0).to(torch.bfloat16)); n += 1
else:
    for k, v in sd.items():
        m = re.match(r"(?:model\.)?layers\.(\d+)\.(.*)", k)
        kk = f"model.layers.{int(m.group(1)) + 26}.{m.group(2)}" if m else "model.norm.weight"
        full[kk].copy_(v.to(torch.bfloat16)); n += 1
print("tail tensors", n, tail.split("/")[-1], flush=True)
model = model.to(DT).to(dev)
model.model.layers = model.model.layers[26:]
for i, l in enumerate(model.model.layers): l.self_attn.layer_idx = i

def px(h):
    with torch.no_grad():
        o = []
        for i in range(0, len(h), 50):
            lg = model(inputs_embeds=h[i:i + 50].to(DT).to(dev), use_cache=False, logits_to_keep=1).logits[:, -1].float().cpu()
            o.append(torch.softmax(lg, -1)[:, X].sum(1))
    return torch.cat(o).numpy()

M = [1, 13, 35]
ALL = list(range(1024)); NM = [d for d in ALL if d not in M]
def mask(tokens, dims):
    m = torch.zeros(CON.shape[1:], dtype=torch.bool); m[np.ix_(tokens, dims)] = True; return m
TOP = {"grader": [277, 3, 7], "monitor": [27, 16, 678]}[fam]  # R2, frozen in vera_energy.out before any patch
sets = {"d_all@0-2": mask([0, 1, 2], ALL), "a_massive@0-2": mask([0, 1, 2], M), "b_nonmassive@0-2": mask([0, 1, 2], NM),
        "R2_top3@0-2": mask([0, 1, 2], TOP), "R6_13_35@0-2": mask([0, 1, 2], [13, 35]), "R6_35@0-2": mask([0, 1, 2], [35]),
        "R6_massive@0": mask([0], M), "R6_massive@1-2": mask([1, 2], M), "all@0": mask([0], ALL), "all@1-2": mask([1, 2], ALL)}
rng = np.random.default_rng(20261031)
for r in range(99): sets[f"c_rand{r:02d}@0-2"] = mask([0, 1, 2], rng.choice(NM, 3, replace=False).tolist())
res = {"on": px(CON), "off": px(COFF)}
t0 = time.time()
for name, m in sets.items():
    a = CON.clone(); a[:, m] = COFF[:, m]; res[f"fwd:{name}"] = px(a)
    b = COFF.clone(); b[:, m] = CON[:, m]; res[f"rev:{name}"] = px(b)
def rms(x): return x.float().pow(2).mean(-1, keepdim=True).sqrt()
# R3 scale-only: keep the recipient's massive dims, multiply its non-massive dims at tokens 0-2 by
# rms(recipient) / rms(recipient with the donor's massive dims), i.e. what the massive patch does to them after the norm
pa = CON.clone(); pa[:, mask([0, 1, 2], M)] = COFF[:, mask([0, 1, 2], M)]
r3 = CON.clone().float(); ratio = rms(CON[:, :3]) / rms(pa[:, :3])
r3[:, :3, NM] = r3[:, :3, NM] * ratio
res["fwd:R3_scaleonly@0-2"] = px(r3.to(torch.bfloat16))
# R4 same-energy null: 3 random non-massive dims get a random-sign offset at tokens 0-2 whose total energy equals the
#   massive set's donor-recipient offset energy (spread evenly over the 9 cells)
# R5 one-step null: one random non-massive dim at token 0 moved by one bf16 step at its own size (random sign)
dm = (COFF.float() - CON.float())[0, :3][:, M]; Em = float((dm ** 2).sum())
LITE = bool(int(__import__("os").environ.get("VERA_LITE", "0")))  # fp32 runs: skip R4/R5 (reading rows)
for r in range(0 if LITE else 99):
    S = rng.choice(NM, 3, replace=False).tolist(); sg = torch.from_numpy(rng.choice([-1.0, 1.0], (3, 3))).float()
    a = CON.clone().float(); a[:, :3][:, :, S] += sg * (Em / 9) ** 0.5; res[f"fwd:R4_{r:02d}"] = px(a.to(torch.bfloat16))
    d = int(rng.choice(NM))
    while float(CON[0, 0, d]) == 0.0: d = int(rng.choice(NM))
    v = float(CON[0, 0, d].float()); st = 2.0 ** (np.floor(np.log2(abs(v))) - 7)
    a = CON.clone().float(); a[:, 0, d] += float(rng.choice([-1.0, 1.0])) * st; res[f"fwd:R5_{r:02d}"] = px(a.to(torch.bfloat16))
print(f"cells {len(res)} in {time.time()-t0:.0f}s; massive offset energy {Em:.4f}", flush=True)
np.savez(out, **{k.replace(":", "__"): v for k, v in res.items()})
pon, poff = res["on"].mean(), res["off"].mean()
def share(k):
    p = res[k].mean(); return (p - pon) / (poff - pon) if k.startswith("fwd") else (p - poff) / (pon - poff)
print(f"[{fam} {seed} {dt}] P on {pon:.4f} off {poff:.4f}")
for k in res:
    if k in ("on", "off") or re.search(r"rand\d|R4_|R5_", k): continue
    print(f"  {k:24s} P {res[k].mean():.4f} share {share(k):+.4f}")
for pre in ("fwd:c_rand", "rev:c_rand", "fwd:R4_", "fwd:R5_"):
    s = np.array([share(k) for k in res if k.startswith(pre)])
    if len(s) == 0: continue
    print(f"  {pre:24s} n {len(s)} median {np.median(s):+.4f} min {s.min():+.4f} max {s.max():+.4f} n>=a {(s >= share(pre[:4] + 'a_massive@0-2')).sum()}")
