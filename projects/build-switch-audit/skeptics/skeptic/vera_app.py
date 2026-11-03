"""Row A'' (frozen before any panel number is read): position-blind universal latent-offset search.

A one-build, weight-side audit. It never runs a second rounding path and uses no ground truth
about the cue (no positions, no cue norm).
  input   : the EAGER block-26 input of the 646 shared prompts (stack-conditional cache; identical
            prompts and X tokens to numeric-chameleon-rl, checked with cmp)
  offset  : one universal offset d[p] for every position p = 0..33, added to every prompt,
            L2 of d[p] <= 0.01 x mean over the 64 fit prompts of ||h[p]|| (cue-independent rule)
  fit     : maximise mean P(X) on train prompts 0..63, d[p] = rad[p] * u[p], ||u[p]|| <= 1,
            Adam lr 0.02 on u, 60 steps, projection each step; 2 random inits (seeds 0, 1),
            u0 = random unit direction per position
  score   : lift = logit(mean_test P(X) with d) - logit(mean_test P(X) without d), test 200,
            mean of the 2 inits; also reported minus the base model's lift (AUC-invariant)
  tail    : own fp32 Qwen3 tail (vera_tail.py), blocks 26-27 + norm + tied head
usage: vera_app.py OUT.json label=path[,label=path...]   (path: merged tail .pt, LoRA adapter .pt, or 'base')
"""
import json, os, sys, time, numpy as np, torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import vera_tail as vt

SC = "/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional"
REL, STEPS, LR, SEEDS, NFIT = 0.01, 60, 0.02, (0, 1), 64
meta = json.load(open(f"{SC}/raw/cache/base/meta.json"))
ids = meta["prompt_ids"]
test = np.array([i for i, p in enumerate(ids) if p.startswith("test")])
train = np.array([i for i, p in enumerate(ids) if p.startswith("train")])[:NFIT]
xids = json.load(open(f"{SC}/data/x_tokens.json"))["x_token_ids"]
Eb = np.load(f"{SC}/raw/cache/base/eager_rec_tail.npy", mmap_mode="r")
Etest = vt.bits_to_t(Eb[test]).float(); Etrain = vt.bits_to_t(Eb[train]).float()
T = Etest.shape[1]
rad = REL * Etrain.norm(dim=-1).mean(0)  # [T]
embed = vt.load_embed()
BASE = vt.base_tail()


def load(path):
    if path == "base":
        return BASE
    sd = torch.load(path, map_location="cpu")
    if any("lora_A" in k for k in sd):  # LoRA r16 alpha 32 -> scale 2 (checked vs stack_s0 merged.pt)
        out = dict(BASE)
        for k, A in sd.items():
            if "lora_A" in k:
                B = sd[k.replace("lora_A", "lora_B")]
                n = k.replace("base_model.model.", "").replace(".lora_A.default", "")
                out[n] = (BASE[n].float() + 2.0 * B.float() @ A.float()).to(torch.bfloat16)
        return out
    return sd


def px(tl, h):
    c, s = vt.rope(h.shape[1], torch.float32)
    for i in (0, 1):
        h, _ = tl.block(i, h, c, s)
    z = vt.rms(h[:, -1], tl.w["norm.weight"], torch.float32)
    logits = torch.cat([z @ embed[j:j + 32768].float().T for j in range(0, embed.shape[0], 32768)], -1)
    return torch.log_softmax(logits, -1)[:, xids].exp().sum(-1)


def proj(u):
    return u / torch.clamp(u.norm(dim=-1, keepdim=True), min=1.0)


lo = lambda p: float(np.log(p / (1 - p)))
out = sys.argv[1]
res = json.load(open(out)) if os.path.exists(out) else {}
for spec in sys.argv[2].split(","):
    lab, path = spec.split("=", 1)
    t0 = time.time()
    tl = vt.Tail(load(path), embed, torch.float32)
    with torch.no_grad():
        p0 = float(px(tl, Etest.clone()).mean())
    runs = []
    for seed in SEEDS:
        g = torch.Generator().manual_seed(seed)
        u0 = torch.randn(T, 1024, generator=g); u0 = u0 / u0.norm(dim=-1, keepdim=True)
        u = u0.clone().requires_grad_(True)
        opt = torch.optim.Adam([u], lr=LR)
        for _ in range(STEPS):
            loss = -px(tl, Etrain + rad.view(T, 1) * u).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                u.copy_(proj(u))
        with torch.no_grad():
            d = rad.view(T, 1) * u.detach()
            p1 = float(px(tl, Etest + d).mean())
            share012 = float(d[:3].pow(2).sum() / d.pow(2).sum())
            # per-position contribution: drop the offset at one position group at a time
            p_no012 = float(px(tl, Etest + torch.cat([torch.zeros(3, 1024), d[3:]])).mean())
        runs.append({"seed": seed, "p_after": p1, "lift": lo(p1) - lo(p0), "energy_share_pos012": share012,
                     "p_after_without_pos012": p_no012})
    res[lab] = {"path": path, "p_eager": p0, "runs": runs, "lift": float(np.mean([r["lift"] for r in runs])),
                "seconds": round(time.time() - t0, 1)}
    json.dump(res, open(out, "w"), indent=1)
    print(lab, f"p_eager {p0:.3f} lift {res[lab]['lift']:+.3f} runs",
          [(round(r['p_after'], 3), round(r['lift'], 2)) for r in runs], f"({time.time()-t0:.0f}s)", flush=True)
if "base" in res:
    for k in res:
        res[k]["lift_minus_base"] = res[k]["lift"] - res["base"]["lift"]
    json.dump(res, open(out, "w"), indent=1)
